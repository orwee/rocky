import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import os

# Si usas la API de OpenAI directamente:
import openai

# Para la memoria de mensajes en Streamlit
from typing import List

########################################################################
#                         FUNCIONES AUXILIARES                         #
########################################################################

def summarize_portfolio(df):
    """
    Recibe un DataFrame con columnas:
      ['chain', 'common_name', 'module', 'token_symbol', 'balance_usd']
    Devuelve un string describiendo sucintamente las posiciones del usuario.
    """
    if df.empty:
        return "El portafolio está vacío o no hay posiciones mayores a \$5."

    summary_lines = []
    total_balance = df['balance_usd'].sum()
    summary_lines.append(f"Balance total estimado: ${format_number(total_balance)}\n")

    # Agrupar por protocolo o como prefieras
    grouped = df.groupby('common_name')['balance_usd'].sum().reset_index()
    summary_lines.append("Resumen por Protocolo:")
    for _, row in grouped.iterrows():
        summary_lines.append(f" - {row['common_name']}: ${format_number(row['balance_usd'])}")

    summary_lines.append("\nPosiciones detalladas (token/protocolo):")
    for idx, row in df.iterrows():
        summary_lines.append(f" • {row['token_symbol']} en {row['common_name']} con balance de ${format_number(row['balance_usd'])}")

    return "\n".join(summary_lines)
    
def get_openai_api_key():
    """
    Obtiene la API key de OpenAI desde Streamlit secrets o,
    si no está guardada ahí, se pide al usuario (texto).

    Ajusta según tu preferencia.
    """
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    return api_key

def format_number(value):
    """Formatea números grandes (ej. 1 millón) con sufijos o comas."""
    if abs(value) >= 1e6:
        return f"{value:,.2f}".rstrip('0').rstrip('.')
    else:
        return f"{value:.6f}".rstrip('0').rstrip('.')

def get_user_defi_positions(address, api_key):
    """
    Llama a la API de Merlin (o la tuya) para obtener posiciones DeFi de un usuario.
    Retorna un objeto JSON con la información o un dict con 'error'.
    """
    base_url = "https://api-v1.mymerlin.io/api/merlin/public/userDeFiPositions/all"
    url = f"{base_url}/{address}"
    headers = {"Authorization": f"{api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def process_defi_data(result):
    """
    Procesa la respuesta de get_user_defi_positions y la convierte en un DataFrame.
    Filtra sólo balances >\$5.
    """
    if not result or not isinstance(result, list):
        return pd.DataFrame(columns=['chain', 'common_name', 'module', 'token_symbol', 'balance_usd'])

    data = []
    for protocol in result:
        chain = str(protocol.get('chain', ''))
        common_name = str(protocol.get('commonName', ''))

        for portfolio in protocol.get('portfolio', []):
            module = str(portfolio.get('module', ''))

            if 'detailed' in portfolio and 'supply' in portfolio['detailed']:
                supply_tokens = portfolio['detailed']['supply']
                if isinstance(supply_tokens, list):
                    if module == 'Liquidity Pool' and len(supply_tokens) >= 2:
                        try:
                            balance_usd_0 = float(supply_tokens[0].get('balanceUSD', 0))
                            balance_usd_1 = float(supply_tokens[1].get('balanceUSD', 0))
                            data.append({
                                'chain': chain,
                                'common_name': common_name,
                                'module': module,
                                'token_symbol': f"{supply_tokens[0].get('tokenSymbol', '')}/{supply_tokens[1].get('tokenSymbol', '')}",
                                'balance_usd': balance_usd_0 + balance_usd_1
                            })
                        except:
                            continue
                    else:
                        for token in supply_tokens:
                            try:
                                data.append({
                                    'chain': chain,
                                    'common_name': common_name,
                                    'module': module,
                                    'token_symbol': str(token.get('tokenSymbol', '')),
                                    'balance_usd': float(token.get('balanceUSD', 0))
                                })
                            except:
                                continue

    if not data:
        return pd.DataFrame(columns=['chain', 'common_name', 'module', 'token_symbol', 'balance_usd'])

    df = pd.DataFrame(data)
    df['balance_usd'] = pd.to_numeric(df['balance_usd'], errors='coerce').fillna(0)
    # Filtrar balances mínimos y redondear
    df = df[df['balance_usd'] > 5]
    df['balance_usd'] = df['balance_usd'].round(6)
    return df

def get_defi_llama_yields():
    """
    Consulta pools de https://yields.llama.fi/pools
    Retorna JSON o dict con 'error'
    """
    url = "https://yields.llama.fi/pools"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

def get_alternatives_for_token(token_symbol, llama_data, n=3):
    """
    Dado un token_symbol y la data de DeFi Llama,
    encuentra pools con APY altos que contengan ese token.
    """
    if not llama_data or 'data' not in llama_data:
        return []
    tokens = token_symbol.split('/')
    alternatives = []
    for pool in llama_data['data']:
        if any(token.upper() in pool['symbol'].upper() for token in tokens):
            alternatives.append({
                'symbol': pool.get('symbol', ''),
                'project': pool.get('project', ''),
                'chain': pool.get('chain', ''),
                'apy': pool.get('apy', 0),
                'tvlUsd': pool.get('tvlUsd', 0)
            })
    # Ordenar por APY descendente
    alternatives.sort(key=lambda x: x['apy'], reverse=True)
    return alternatives[:n]

def generate_investment_analysis(current_position, alternatives, api_key):
    """
    Realiza una llamada a la API de OpenAI para generar un análisis breve
    comparando la posición actual vs. las alternativas.
    """
    if not api_key:
        return "Error: Falta la OpenAI API key."

    openai.api_key = api_key

    prompt = f"""
    Eres un asesor DeFi experto.
    Analiza brevemente esta posición y posibles alternativas:

    Posición actual:
    - Token: {current_position['token_symbol']}
    - Protocolo: {current_position['common_name']}
    - Balance USD: ${format_number(current_position['balance_usd'])}

    Alternativas disponibles:
    {chr(10).join([f"- {alt['project']} en {alt['chain']}: {alt['symbol']} (APY: {alt['apy']:.2f}%, TVL: ${format_number(alt['tvlUsd'])})" for alt in alternatives])}

    Da un comentario conciso (máx 100 palabras) y una recomendación final.
    """

    try:
        # Llamada al endpoint Chat GPT3.5 o GPT-4, ajusta según tu preferencia
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asesor DeFi experto y muy conciso."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error al generar el análisis: {e}"

########################################################################
#                           LÓGICA DE CHAT                             #
########################################################################

def init_chat_history():
    """
    Inicializa (o recupera) el historial de chat en session_state.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente DeFi. Pregúntame sobre tu portafolio o alternativas de inversión."}
        ]

def render_chat():
    # Mostrar el historial anterior
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Escribe tu pregunta o solicitud aquí...")
    if user_input:
        # Agregar mensaje del usuario
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Preparar la llamada a OpenAI
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            ai_response = "Por favor, agrega tu OpenAI API key para continuar."
        else:
            openai.api_key = openai_api_key

            # 1. Construir lista de mensajes para mandar a ChatCompletion
            messages_for_openai = []

            # (a) Mensaje de sistema/bienvenida
            messages_for_openai.append({
                "role": "system",
                "content": (
                    "Actúa como un asesor experto en DeFi. "
                    "A continuación tienes un resumen del portafolio del usuario. Úsalo para responder de forma contextual.\n\n"
                    f"{st.session_state.get('portfolio_summary', 'No hay resumen de portafolio disponible.')}"
                )
            })

            # (b) Luego, añade los mensajes previos del chat
            messages_for_openai.extend(st.session_state["messages"])

            # 2. Finalmente, llama a la API
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages_for_openai
                )
                ai_response = completion["choices"][0]["message"]["content"]
            except Exception as e:
                ai_response = f"Error al generar respuesta: {e}"

        # Guardar la respuesta en la historia y mostrarla
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})
        st.chat_message("assistant").write(ai_response)

########################################################################
#                              APP MAIN                                #
########################################################################

def main():
    st.set_page_config(page_title="Mi Agente DeFi", layout="wide")

    # Columnas: izquierda para portafolio, derecha para chat
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.title("Resumen de Portafolio DeFi")

        st.sidebar.header("Ajustes")

        # Pedir wallet y una API de Merlin
        wallet_address = st.sidebar.text_input("Wallet Address")
        merlin_api_key = st.sidebar.text_input("Merlin API Key", value="", type="password")

        if "analyze" not in st.session_state:
            st.session_state["analyze"] = False

        if st.sidebar.button("Analizar Portafolio"):
            st.session_state["analyze"] = True

        if st.session_state["analyze"] and wallet_address and merlin_api_key:
            # Obtener datos del usuario
            result = get_user_defi_positions(wallet_address, merlin_api_key)
            if 'error' not in result:
                df = process_defi_data(result)

                if not df.empty:
                    # Mostrar DF
                    st.subheader("Tus Posiciones DeFi")
                    df_display = df.copy()
                    df_display["balance_usd"] = df_display["balance_usd"].apply(lambda x: f"${format_number(x)}")
                    st.dataframe(df_display, use_container_width=True)

                    # Some graphs
                    if df['balance_usd'].sum() > 0:
                        st.subheader("Distribución de Balance USD")
                        c1, c2 = st.columns(2)

                        with c1:
                            df_grouped = df.groupby(['token_symbol','common_name'])['balance_usd'].sum().reset_index()
                            fig = px.pie(
                                df_grouped,
                                values='balance_usd',
                                names=df_grouped.apply(lambda x: f"{x['token_symbol']} ({x['common_name']})", axis=1),
                                title='Por Token/Protocolo'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with c2:
                            df_grouped_mod = df.groupby('module')['balance_usd'].sum().reset_index()
                            fig2 = px.pie(
                                df_grouped_mod,
                                values='balance_usd',
                                names='module',
                                title='Por Módulo'
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        # Métricas básicas:
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Balance USD", f"${format_number(df['balance_usd'].sum())}")
                        col_b.metric("Núm. de Protocolos", len(df['common_name'].unique()))
                        col_c.metric("Núm. de Posiciones", len(df))

                    #####################
                    # Alternativas
                    st.subheader("Alternativas de Inversión DeFi")
                    llama_data = get_defi_llama_yields()
                    st.session_state["portfolio_summary"] = summarize_portfolio(df)
                    
                    if 'error' not in llama_data:
                        for idx, row in df.iterrows():
                            with st.expander(f"{row['token_symbol']} en {row['common_name']}"):
                                alternatives = get_alternatives_for_token(row['token_symbol'], llama_data)
                                if alternatives:
                                    df_alt = pd.DataFrame(alternatives)
                                    df_alt['apy'] = df_alt['apy'].apply(lambda x: f"{x:.2f}%")
                                    df_alt['tvlUsd'] = df_alt['tvlUsd'].apply(lambda x: f"${format_number(x)}")
                                    st.dataframe(df_alt, use_container_width=True)

                                    # Análisis GPT
                                    openai_key = get_openai_api_key()
                                    analysis = generate_investment_analysis(row, alternatives, openai_key)
                                    st.markdown(f"**Análisis breve:** {analysis}")
                                else:
                                    st.info("No se encontraron alternativas.")
                    else:
                        st.warning("No se pudo consultar DeFiLlama.")
                else:
                    st.warning("No se encontraron posiciones > \$5 en esta wallet.")
            else:
                st.error(result["error"])

    # Columna derecha -> Chat
    with col_right:
        st.title("Chat DeFi")
        init_chat_history()
        render_chat()

if __name__ == "__main__":
    main()
