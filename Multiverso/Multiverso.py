import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

# Configuração da Página
st.set_page_config(
    page_title="Simulador Multiversal N-D",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Simulador e Visualizador Multiversal (N-Dimensões)")
st.caption("Cosmologia Relativística baseada na Sequência e Constante de Tribonacci")

# Constante de Tribonacci
ALPHA = 1.839286755214161

# --- ESTADO INICIAL (SESSION STATE) ---
if "dimensions" not in st.session_state:
    st.session_state.dimensions = ["w (Fase)", "η (Profundidade)"]

if "lambda_val" not in st.session_state:
    st.session_state.lambda_val = 31.0

if "planets" not in st.session_state:
    st.session_state.planets = [
        {"Nome": "Origem (Singularidade)", "n": 1, "w (Fase)": 0.0, "η (Profundidade)": 0.0, "Cor": "#ff0066"},
        {"Nome": "Plano Infernal", "n": 3, "w (Fase)": 0.0, "η (Profundidade)": 0.0, "Cor": "#ff3300"},
        {"Nome": "Plano Umbral", "n": 5, "w (Fase)": 2.0, "η (Profundidade)": 5.0, "Cor": "#6600cc"},
        {"Nome": "Plano Astral", "n": 7, "w (Fase)": -3.0, "η (Profundidade)": 2.0, "Cor": "#00cccc"},
        {"Nome": "Plano Material", "n": 10, "w (Fase)": 0.0, "η (Profundidade)": 0.0, "Cor": "#33cc33"},
        {"Nome": "Plano Etéreo", "n": 12, "w (Fase)": 5.0, "η (Profundidade)": 1.0, "Cor": "#9933ff"},
    ]

# --- PAINEL LATERAL: CONTROLES & GESTÃO DE ARQUIVOS ---
st.sidebar.header("💾 Salvar & Compartilhar Cosmologia")

export_data = {
    "lambda_val": float(st.session_state.lambda_val),
    "dimensions": st.session_state.dimensions,
    "planets": st.session_state.planets
}
json_string = json.dumps(export_data, indent=4, ensure_ascii=False)

st.sidebar.download_button(
    label="📥 Baixar Cosmologia Atual (JSON)",
    data=json_string,
    file_name="multiverso_config.json",
    mime="application/json",
    help="Baixa a configuração completa de planos e dimensões para guardar ou compartilhar com outros."
)

uploaded_file = st.sidebar.file_uploader(
    "📂 Carregar Cosmologia (JSON):", 
    type=["json"],
    help="Carregue um arquivo .json salvo previamente para restaurar os planos."
)

if uploaded_file is not None:
    try:
        loaded_data = json.load(uploaded_file)
        if st.sidebar.button("⚙️ Aplicar Configuração Carregada"):
            st.session_state.lambda_val = loaded_data.get("lambda_val", 31.0)
            st.session_state.dimensions = loaded_data.get("dimensions", ["w (Fase)", "η (Profundidade)"])
            st.session_state.planets = loaded_data.get("planets", [])
            st.sidebar.success("Cosmologia carregada com sucesso!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Erro ao ler o arquivo JSON: {e}")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parâmetros Globais")
st.session_state.lambda_val = st.sidebar.number_input(
    "Constante Lambda (λ)", 
    value=float(st.session_state.lambda_val), 
    min_value=0.1, 
    step=1.0,
    help="Define a escala de influência da Origem. Menor = transição de tempo mais brusca."
)

st.sidebar.markdown("---")
st.sidebar.header("🌀 Gerenciador de Dimensões")

new_dim = st.sidebar.text_input("Nova Dimensão Extra:")
if st.sidebar.button("➕ Adicionar Dimensão"):
    if new_dim and new_dim not in st.session_state.dimensions:
        st.session_state.dimensions.append(new_dim)
        for p in st.session_state.planets:
            p[new_dim] = 0.0
        st.rerun()

dims_to_remove = st.sidebar.multiselect("Remover Dimensões Extras:", st.session_state.dimensions)
if st.sidebar.button("🗑️ Remover Dimensões Selecionadas"):
    for d in dims_to_remove:
        st.session_state.dimensions.remove(d)
        for p in st.session_state.planets:
            p.pop(d, None)
    st.rerun()

# --- CÁLCULOS MATEMÁTICOS ---
def tribonacci_coords(n):
    theta = (n - 1) * (2 * np.pi / ALPHA)
    r_spatial = np.power(ALPHA, (n - 1) / 3.0)
    x = r_spatial * np.cos(theta)
    y = r_spatial * np.sin(theta)
    z = (n - 1) * 2.5
    return x, y, z, theta

def calculate_r_and_psi(planet, extra_dims, lambda_val):
    x0, y0, z0, theta = tribonacci_coords(planet["n"])
    
    dx, dy, dz = 0.0, 0.0, 0.0
    
    if "w (Fase)" in extra_dims:
        w_val = float(planet.get("w (Fase)", 0.0))
        dx += w_val * np.cos(theta)
        dy += w_val * np.sin(theta)
        
    if "η (Profundidade)" in extra_dims:
        eta_val = float(planet.get("η (Profundidade)", 0.0))
        dz += eta_val
        
    other_dims_sum = 0.0
    for dim in extra_dims:
        if dim not in ["w (Fase)", "η (Profundidade)"]:
            other_dims_sum += float(planet.get(dim, 0.0))

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz + other_dims_sum

    sum_sq = x**2 + y**2 + z**2
    for dim in extra_dims:
        sum_sq += float(planet.get(dim, 0.0))**2
    r = np.sqrt(sum_sq)
    psi = r / (r + lambda_val)
    return r, psi, x, y, z

planet_data = []
for p in st.session_state.planets:
    r, psi, x, y, z = calculate_r_and_psi(p, st.session_state.dimensions, st.session_state.lambda_val)
    
    w_val = float(p.get("w (Fase)", 0.0))
    eta_val = float(p.get("η (Profundidade)", 0.0))
    fora_curva = "Sim" if (w_val != 0.0 or eta_val != 0.0) else "Não"
    
    item = {**p, "Fora da Curva": fora_curva, "r (Distância 5D)": r, "Ψ (Velocidade Tempo)": psi, "X": x, "Y": y, "Z": z}
    planet_data.append(item)

df_planets = pd.DataFrame(planet_data)

# --- ABAS DA INTERFACE PRINCIPAL ---
tab1, tab2, tab3 = st.tabs(["📊 Calculador Temporal & Planos", "🌌 Visualizador 3D & Fatiamento", "⚡ Sobreposições (Overlap)"])

# === ABA 1: CALCULADOR DE TEMPO & PLANOS ===
with tab1:
    with st.expander("📖 Guia de Termos & Variáveis do Mestre (Clique para abrir/fechar)"):
        st.markdown("""
        * **Constante Lambda ($\lambda$):** O raio de escala do multiverso. Define a intensidade da Origem: quanto menor, mais abrupta é a desaceleração do tempo perto do centro.
        * **Índice de Tribonacci ($n$):** A posição base do plano na "escada" principal da espiral de mundos.
        * **Desvio de Fase ($w$) & Profundidade ($\eta$):** Variáveis que deslocam automaticamente o plano para fora da curva principal da espiral no espaço 3D.
        * **Cor:** A cor visual do plano no mapa tridimensional e exibida como bloco na tabela.
        * **Distância Radial ($r$):** A distância euclidiana total do plano até a Origem no hiperespaço $N$-dimensional.
        * **Velocidade Temporal ($\Psi$):** O fator de fluxo temporal do plano (varia de 0 [estase total na origem] a 1 [tempo infinito/livre]).
        """)

    st.subheader("⏱️ Cálculo de Dilatação Temporal")
    
    planets_list = df_planets["Nome"].tolist()
    
    if len(planets_list) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ref_name = st.selectbox("Plano de Referência:", planets_list, index=min(4, len(planets_list)-1))
        with col2:
            dest_name = st.selectbox("Plano de Destino:", planets_list, index=min(1, len(planets_list)-1))
        with col3:
            time_unit = st.selectbox("Unidade:", ["Minutos", "Horas", "Dias", "Semanas"], index=1)
        with col4:
            input_time = st.number_input("Tempo no Destino:", value=1.0, min_value=0.0, step=0.5)

        multipliers = {"Minutos": 1/60, "Horas": 1.0, "Dias": 24.0, "Semanas": 24.0*7.0}
        input_time_hours = input_time * multipliers[time_unit]

        ref_row = df_planets[df_planets["Nome"] == ref_name].iloc[0]
        dest_row = df_planets[df_planets["Nome"] == dest_name].iloc[0]

        psi_ref = ref_row["Ψ (Velocidade Tempo)"]
        psi_dest = dest_row["Ψ (Velocidade Tempo)"]
        
        ratio = psi_dest / psi_ref
        time_in_ref_hours = input_time_hours / ratio if ratio > 0 else 0

        if time_in_ref_hours >= 24*7:
            res_friendly = f"{time_in_ref_hours / (24*7):.2f} Semanas"
        elif time_in_ref_hours >= 24:
            res_friendly = f"{time_in_ref_hours / 24:.2f} Dias"
        elif time_in_ref_hours >= 1:
            res_friendly = f"{time_in_ref_hours:.2f} Horas"
        else:
            res_friendly = f"{time_in_ref_hours * 60:.2f} Minutos"

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(
                label=f"Proporção de Fluxo ({dest_name} / {ref_name})", 
                value=f"{ratio:.4f}",
                delta=f"1 unidade em {dest_name} = {1/ratio:.2f} na mesma unidade em {ref_name}" if ratio > 0 else "Estase Total"
            )
        with res_col2:
            st.metric(
                label=f"Tempo passado em '{ref_name}'", 
                value=res_friendly,
                delta=f"Exatos {time_in_ref_hours:.2f} Horas"
            )
    else:
        st.warning("Nenhum plano cadastrado. Adicione um plano abaixo para começar.")

    st.markdown("---")
    st.subheader("📋 Gerenciamento de Planos de Existência")

    # --- AVISO CONDICIONAL DE PLANOS DE EXEMPLO ---
    default_names = {
        "Origem (Singularidade)", "Plano Infernal", "Plano Umbral", 
        "Plano Astral", "Plano Material", "Plano Etéreo"
    }
    current_names = {p["Nome"] for p in st.session_state.planets}

    if current_names == default_names:
        st.info("💡 **Nota para o Mestre:** Os planos listados abaixo são apenas **exemplos padrão** ilustrativos para teste. Assim que você editar, excluir ou adicionar novos planos, este aviso sumirá automaticamente!")

    col_add, col_edit, col_del = st.columns(3)
    
    # ➕ ADICIONAR PLANO
    with col_add:
        with st.expander("➕ Adicionar Novo Plano"):
            with st.form("add_planet_form"):
                new_name = st.text_input("Nome do Plano:")
                new_color = st.color_picker("Cor do Plano:", value="#00ffcc")
                new_n = st.number_input("Índice Tribonacci (n):", value=8, min_value=1, max_value=30)

                extra_vals = {}
                for dim in st.session_state.dimensions:
                    extra_vals[dim] = st.number_input(f"Valor de {dim}:", value=0.0)
                
                submitted = st.form_submit_button("Criar Plano")
                if submitted and new_name:
                    new_p = {
                        "Nome": new_name, 
                        "n": int(new_n), 
                        "Cor": new_color,
                        **extra_vals
                    }
                    st.session_state.planets.append(new_p)
                    st.rerun()

    # ✏️ EDITAR PLANO EXISTENTE
    with col_edit:
        with st.expander("✏️ Editar Plano Existente"):
            if len(st.session_state.planets) > 0:
                selected_edit_name = st.selectbox(
                    "Selecione o plano para editar:",
                    options=[p["Nome"] for p in st.session_state.planets]
                )
                
                edit_index = next(i for i, p in enumerate(st.session_state.planets) if p["Nome"] == selected_edit_name)
                target_p = st.session_state.planets[edit_index]
                
                with st.form("edit_planet_form"):
                    edit_name = st.text_input("Novo Nome:", value=target_p["Nome"])
                    edit_color = st.color_picker("Nova Cor:", value=target_p.get("Cor", "#00ffcc"))
                    edit_n = st.number_input("Índice Tribonacci (n):", value=int(target_p.get("n", 8)), min_value=1, max_value=30)

                    edit_extra_vals = {}
                    for dim in st.session_state.dimensions:
                        edit_extra_vals[dim] = st.number_input(
                            f"Valor de {dim}:", 
                            value=float(target_p.get(dim, 0.0))
                        )
                    
                    submitted_edit = st.form_submit_button("Salvar Alterações")
                    if submitted_edit and edit_name:
                        st.session_state.planets[edit_index] = {
                            "Nome": edit_name,
                            "n": int(edit_n),
                            "Cor": edit_color,
                            **edit_extra_vals
                        }
                        st.rerun()
            else:
                st.info("Nenhum plano disponível para edição.")

    # 🗑️ REMOVER PLANO
    with col_del:
        with st.expander("🗑️ Remover Planos Existentes"):
            with st.form("del_planet_form"):
                planets_to_remove = st.multiselect(
                    "Selecione os planos que deseja apagar:",
                    options=[p["Nome"] for p in st.session_state.planets]
                )
                submitted_del = st.form_submit_button("Excluir Selecionados")
                if submitted_del and planets_to_remove:
                    st.session_state.planets = [p for p in st.session_state.planets if p["Nome"] not in planets_to_remove]
                    st.rerun()

    if len(df_planets) > 0:
        def color_block(val):
            return f'background-color: {val}; color: {val};'

        display_cols = ["Nome", "Fora da Curva", "Cor", "r (Distância 5D)", "Ψ (Velocidade Tempo)", "X", "Y", "Z"] + st.session_state.dimensions
        styled_table = df_planets[display_cols].style.map(color_block, subset=['Cor'])
        st.dataframe(styled_table, use_container_width=True)

# === ABA 2: VISUALIZADOR 3D & FATIAMENTO ===
with tab2:
    with st.expander("📖 Guia de Fatiamento Hiperdimensional (Como usar)"):
        st.markdown("""
        * **Fatias de Dimensões Extras ($w, \eta, \dots$):** Estes controles selecionam uma "coordenada exata" nas dimensões superiores para fatiar o hiperespaço.
        * **Tolerância de Corte (Espessura):** Define o quão "larga" é a fatia. Planos dentro dessa margem aparecem **opacos e destacados** em 3D.
        * **Planos Fora da Curva:** Mundos cujas variáveis de fase ($w$) ou profundidade ($\eta$) são diferentes de zero ganham automaticamente um formato de **diamante (⭐)** no gráfico, indicando que se afastaram da espiral base.
        """)

    st.subheader("🎨 Espiral de Tribonacci & Fatiamento Hiperdimensional")
    st.caption("Ajuste os controles das dimensões extras para 'fatiar' o hiperespaço e ver quais planos permanecem na fatia.")

    if len(df_planets) > 0:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            color_mode = st.radio(
                "Modo de Coloração 3D:", 
                ["Cores Personalizadas dos Planos", "Gradiente por Velocidade do Tempo (Ψ)"],
                horizontal=True
            )

        slice_filters = {}
        slice_cols = st.columns(max(1, len(st.session_state.dimensions)))
        for idx, dim in enumerate(st.session_state.dimensions):
            with slice_cols[idx]:
                slice_filters[dim] = st.slider(f"Fatia de {dim}:", -20.0, 20.0, 0.0, step=0.5)

        tolerance = st.slider("Tolerância de Corte (Espessura da Fatia):", 0.5, 10.0, 3.0)

        n_dense = np.linspace(1, max(df_planets["n"].max() + 2, 15), 300)
        theta_dense = (n_dense - 1) * (2 * np.pi / ALPHA)
        r_dense = np.power(ALPHA, (n_dense - 1) / 3.0)
        x_dense = r_dense * np.cos(theta_dense)
        y_dense = r_dense * np.sin(theta_dense)
        z_dense = (n_dense - 1) * 2.5

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_dense, y=y_dense, z=z_dense,
            mode='lines',
            line=dict(color='rgba(150, 100, 255, 0.5)', width=4),
            name='Espiral de Tribonacci'
        ))

        for idx, row in df_planets.iterrows():
            dist_slice = np.sqrt(sum([(row[dim] - slice_filters[dim])**2 for dim in st.session_state.dimensions]))
            in_slice = dist_slice <= tolerance
            
            opacity = 1.0 if in_slice else 0.15
            
            if color_mode == "Gradiente por Velocidade do Tempo (Ψ)":
                color = f"rgb({int(255*(1-row['Ψ (Velocidade Tempo)']))}, {int(255*row['Ψ (Velocidade Tempo)'])}, 200)"
            else:
                color = row.get("Cor", "#00ffcc")

            is_off_curve = (row["Fora da Curva"] == "Sim")

            fig.add_trace(go.Scatter3d(
                x=[row["X"]], y=[row["Y"]], z=[row["Z"]],
                mode='markers+text',
                marker=dict(
                    size=14 if in_slice else 7,
                    color=color,
                    opacity=opacity,
                    symbol='diamond' if is_off_curve else 'circle',
                    line=dict(width=1, color='white')
                ),
                text=[f"⭐ {row['Nome']}" if is_off_curve else row["Nome"]],
                textposition="top center",
                name=row["Nome"],
                hovertemplate=(
                    f"<b>{row['Nome']}</b> (Fora da Curva)<br>" if is_off_curve else f"<b>{row['Nome']}</b><br>"
                ) + f"Ψ (Tempo): {row['Ψ (Velocidade Tempo)']:.3f}<br>Distância 5D: {row['r (Distância 5D)']:.2f}<br>"
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X (Tribonacci)',
                yaxis_title='Y (Tribonacci)',
                zaxis_title='Z (Altura)',
                aspectmode='data',
                bgcolor='rgb(15, 15, 25)'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor='rgba(0,0,0,0)',
            height=650,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Adicione pelo menos um plano na aba anterior para visualizar o multiverso em 3D.")

# === ABA 3: SOBREPOSIÇÕES (OVERLAP) ===
with tab3:
    st.subheader("⚡ Detector de Sangramento Dimensional (Dimensional Overlap)")
    st.write("Procura por planos que estão próximos no espaço 3D $(X,Y,Z)$, mas separados nas dimensões extras.")

    if len(df_planets) > 1:
        dist_3d_threshold = st.slider("Distância Máxima 3D para Sobreposição:", 1.0, 30.0, 10.0)
        
        overlaps = []
        n_planets = len(planet_data)
        for i in range(n_planets):
            for j in range(i + 1, n_planets):
                p1 = planet_data[i]
                p2 = planet_data[j]
                
                d3d = np.sqrt((p1["X"]-p2["X"])**2 + (p1["Y"]-p2["Y"])**2 + (p1["Z"]-p2["Z"])**2)
                if d3d <= dist_3d_threshold:
                    d_extra = np.sqrt(sum([(p1.get(dim,0) - p2.get(dim,0))**2 for dim in st.session_state.dimensions]))
                    overlaps.append({
                        "Plano A": p1["Nome"],
                        "Plano B": p2["Nome"],
                        "Distância 3D": d3d,
                        "Afastamento Hiperdimensional": d_extra,
                        "Diferença Temporal (ΔΨ)": abs(p1["Ψ (Velocidade Tempo)"] - p2["Ψ (Velocidade Tempo)"])
                    })

        if overlaps:
            st.dataframe(pd.DataFrame(overlaps), use_container_width=True)
        else:
            st.info("Nenhuma sobreposição detectada com o limiar atual.")
    else:
        st.info("Cadastre pelo menos dois planos para analisar sobreposições.")
