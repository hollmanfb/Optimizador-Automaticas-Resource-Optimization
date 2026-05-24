import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (WORKLOAD Y METRIZ DE DISTANCIAS EN METROS)
# -------------------------------------------------------------------------
MAX_SATURACION = 0.97  
META_SATURACION = 0.90
HISTORICO_ASIGNACIONES = "historico_planes.csv"
MEMORIA_ML = "memoria_aprendizaje.csv"

WORKLOAD_MAESTRO = {
    "902": 0.3712, "903": 0.3437, "904": 0.3016, "905": 0.3218,
    "906": 0.3289, "907": 0.3217, "911": 0.1821, "916": 0.3868,
    "917": 1.0000, "922": 0.5321, "923": 0.6995, "924": 1.0000,
    "925": 0.3356, "926": 0.3361, "927": 0.5300, "928": 0.6735
}

MATRIZ_DISTANCIAS = {
    "902": {"902":0, "903":7, "904":23, "905":9, "906":4, "907":8, "911":25, "916":21, "917":41, "922":21, "923":27, "924":15, "925":12, "926":26, "927":6, "928":40},
    "903": {"902":7, "903":0, "904":27, "905":10, "906":8, "907":8.5, "911":20, "916":26, "917":48, "922":21, "923":26, "924":12, "925":15, "926":20, "927":11, "928":47},
    "904": {"902":23, "903":27, "904":0, "905":39, "906":26, "907":29, "911":44, "916":3, "917":27, "922":43, "923":13, "924":24, "925":42, "926":45, "927":15, "928":21},
    "905": {"902":9, "903":10, "904":39, "905":0, "906":12, "907":10, "911":9, "916":28, "917":50, "922":12, "923":28, "924":18, "925":3, "926":6, "927":18, "928":35},
    "906": {"902":4, "903":8, "904":26, "905":12, "906":0, "907":2, "911":20, "916":23, "917":49, "922":19, "923":36, "924":20, "925":18, "926":21, "927":12, "928":37},
    "907": {"902":8, "903":8.5, "904":29, "905":10, "906":2, "907":0, "911":18, "916":25, "917":51, "922":17, "923":38, "924":22, "925":16, "926":19, "927":14, "928":39},
    "911": {"902":25, "903":20, "904":44, "905":9, "906":20, "907":18, "911":0, "916":37, "917":59, "922":1, "923":37, "924":27, "925":3, "926":1, "927":27, "928":44},
    "916": {"902":21, "903":26, "904":3, "905":28, "906":23, "907":25, "911":37, "916":0, "917":27, "922":40, "923":10, "924":21, "925":39, "926":42, "927":12, "928":26},
    "917": {"902":41, "903":48, "904":23, "905":50, "906":49, "907":51, "911":59, "916":27, "917":0, "922":70, "923":38, "924":74, "925":78, "926":80, "927":47, "928":16},
    "922": {"902":21, "903":21, "904":43, "905":12, "906":19, "907":17, "911":1, "916":40, "917":70, "922":0, "923":50, "924":36, "925":10, "926":13, "927":37, "928":62},
    "923": {"902":27, "903":26, "904":13, "905":28, "906":36, "907":38, "911":37, "916":10, "917":38, "922":50, "923":0, "924":19, "925":42, "926":46, "927":22, "928":46},
    "924": {"902":15, "903":12, "904":24, "905":18, "906":20, "907":22, "911":27, "916":21, "917":74, "922":36, "923":19, "924":0, "925":24, "926":28, "927":20, "928":45},
    "925": {"902":12, "903":15, "904":42, "905":3, "906":18, "907":16, "911":3, "916":39, "917":78, "922":10, "923":42, "924":24, "925":0, "926":3, "927":35, "928":62},
    "926": {"902":26, "903":20, "904":45, "905":6, "906":21, "907":19, "911":1, "916":42, "917":80, "922":13, "923":46, "924":28, "925":3, "926":0, "927":38, "928":66},
    "927": {"902":6, "903":11, "904":15, "905":18, "906":12, "907":14, "911":27, "916":12, "917":47, "922":37, "923":22, "924":20, "925":35, "926":38, "927":0, "928":25},
    "928": {"902":40, "903":40, "904":21, "905":35, "906":37, "907":39, "911":44, "916":26, "917":16, "922":62, "923":46, "924":45, "925":62, "926":66, "927":25, "928":0}
}

def cargar_penalizaciones_ml():
    if os.path.exists(MEMORIA_ML):
        try:
            df = pd.read_csv(MEMORIA_ML)
            df_err = df[df["motivo"].str.contains("distancias|coherencia", case=False, na=False)]
            return {str(row["maquina"]): 40.0 for _, row in df_err.iterrows()}
        except: pass
    return {}

def registrar_evento_ml(maquina, motivo, operario):
    nuevo = pd.DataFrame([{"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "maquina": maquina, "motivo": motivo, "operario": operario}])
    nuevo.to_csv(MEMORIA_ML, mode='a', header=not os.path.exists(MEMORIA_ML), index=False)

# -------------------------------------------------------------------------
# 2. ALGORITMO OPTIMIZADO CON TOPE ESTRICTO DEL 97%
# -------------------------------------------------------------------------
def optimizar_asignacion(maquinas_activas, asignaciones_manuales, prioridades):
    operarios = {}
    maquinas_por_asignar = []
    barreras_ml = cargar_penalizaciones_ml()

    for m in maquinas_activas:
        carga_m = WORKLOAD_MAESTRO.get(m, 0)
        if m in asignaciones_manuales:
            op_id = asignaciones_manuales[m]
            if op_id not in operarios: operarios[op_id] = {"maquinas": [], "carga_total": 0.0}
            operarios[op_id]["maquinas"].append(m)
            operarios[op_id]["carga_total"] += carga_m
        elif carga_m >= 1.00:
            operarios[f"Operario Dedicado {m}"] = {"maquinas": [m], "carga_total": carga_m}
        else:
            maquinas_por_asignar.append(m)

    maquinas_por_asignar.sort(key=lambda x: (prioridades.get(x, 2), -WORKLOAD_MAESTRO.get(x, 0)))
    nuevo_op_idx = 1

    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        while op_actual in operarios:
            nuevo_op_idx += 1
            op_actual = f"Operario {nuevo_op_idx}"

        operarios[op_actual] = {"maquinas": [], "carga_total": 0.0}
        maquina_pivote = maquinas_por_asignar.pop(0)
        operarios[op_actual]["maquinas"].append(maquina_pivote)
        operarios[op_actual]["carga_total"] += WORKLOAD_MAESTRO[maquina_pivote]

        while len(maquinas_por_asignar) > 0:
            candidatas = []
            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO[m]
                if operarios[op_actual]["carga_total"] + carga_m <= MAX_SATURACION:
                    dist_base = np.mean([MATRIZ_DISTANCIAS.get(m, {}).get(y, 15.0) for y in operarios[op_actual]["maquinas"]])
                    candidatas.append((m, dist_base + barreras_ml.get(m, 0.0), carga_m))

            if not candidatas: break
            candidatas.sort(key=lambda x: x[1])
            mejor_m, _, mejor_c = candidatas[0]
            
            operarios[op_actual]["maquinas"].append(mejor_m)
            operarios[op_actual]["carga_total"] += mejor_c
            maquinas_por_asignar.remove(mejor_m)
        nuevo_op_idx += 1

    return operarios

# -------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA CORREGIDA (BLINDADA CONTRA ERRORES DE CSS)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador Corporativo")

# Inyección limpia de CSS sin saltos de línea conflictivos usando un string limpio
estilo_limpio = "<style>.stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #1d3557; box-shadow: 0 2px 4px rgba(0,0,0,0.05); } h1, h2, h3 { color: #1d3557 !important; font-family: 'Arial Black', Gadget, sans-serif; }</style>"
st.markdown(estilo_limpio, unsafe_with_html=True)

st.title("🏭 Planificación y Balanceo Dinámico de Cargas")
st.markdown("---")

# Capturar las asignaciones en sesión para mantener el estado entre ejecuciones
if "maquinas_activas" not in st.session_state:
    st.session_state.maquinas_activas = ["927", "902", "922", "911", "905", "907", "903", "923", "924"]

if "asignaciones_manuales" not in st.session_state:
    st.session_state.asignaciones_manuales = {}

if "prioridades" not in st.session_state:
    st.session_state.prioridades = {m: 2 for m in WORKLOAD_MAESTRO.keys()}

# REQUERIMIENTO CENTRAL: SE DESPLIEGAN PRIMERO LOS RESULTADOS GENERADOS
st.header("🚀 1. Propuesta Automática de Distribución del Turno")

if st.session_state.maquinas_activas:
    # Ejecutar la optimización leyendo el estado persistente
    resultado = optimizar_asignacion(
        st.session_state.maquinas_activas, 
        st.session_state.asignaciones_manuales, 
        st.session_state.prioridades
    )
    resultado = {k: v for k, v in resultado.items() if len(v["maquinas"]) > 0}
    
    # KPIs visuales superiores
    k1, k2, k3 = st.columns(3)
    k1.metric("👤 Operarios Requeridos", len(resultado))
    k2.metric("🏭 Inyectoras en Marcha", len(st.session_state.maquinas_activas))
    c_med = np.mean([v["carga_total"] for v in resultado.values()]) * 100
    k3.metric("📊 Saturación Media", f"{c_med:.1f}%")

    st.write(" ")
    cols_res = st.columns(min(len(resultado), 4))
    
    # Construcción estructurada del Layout HTML para la impresión impecable
    html_print = "<html><head><style>body { font-family: Arial, sans-serif; color: #333; margin: 20px; } .header { border-bottom: 3px solid #1d3557; padding-bottom: 10px; margin-bottom: 20px; } .title { font-size: 20pt; font-weight: bold; color: #1d3557; } .card { border: 1px solid #cbd5e1; border-radius: 6px; margin-bottom: 15px; background: #f8fafc; } .card-h { background: #1d3557; color: white; padding: 10px; font-weight: bold; font-size: 12pt; } .card-b { padding: 12px; } .badge { background: #e63946; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9pt; float: right; }</style></head><body><div class='header'><div class='title'>REPARTO DE OPERARIOS EN PLANTA</div><div>Estrategia Dinámica Balanceada</div></div>"

    for idx, (operario, datos) in enumerate(sorted(resultado.items())):
        sat_p = datos['carga_total'] * 100
        color_borde = "#e63946" if sat_p > 97.0 else "#1d3557"
        
        html_print += f"<div class='card'><div class='card-h'>{operario} <span class='badge'>{sat_p:.1f}% Carga</span></div><div class='card-b'><ul>"
        
        with cols_res[idx % 4]:
            st.markdown(f"<div style='background-color: #f8fafc; border: 1px solid #cbd5e1; border-top: 5px solid {color_borde}; padding: 15px; border-radius: 6px; margin-bottom: 10px;'><h4 style='margin:0; color:#1d3557;'>👤 {operario}</h4><p style='margin:5px 0; font-size:15px;'><b>Carga:</b> <code style='color:#e63946;'>{sat_p:.1f}%</code></p><hr style='margin:8px 0; border:0; border-top:1px solid #e2e8f0;'>", unsafe_with_html=True)
            
            for m in datos["maquinas"]:
                st.write(f"• **Máq. {m}** ({WORKLOAD_MAESTRO[m]*100:.1f}%)")
                html_print += f"<li>Máquina {m} ({WORKLOAD_MAESTRO[m]*100:.1f}% Workload)</li>"
            st.markdown("</div>", unsafe_with_html=True)
        html_print += "</ul></div></div>"
    html_print += "</body></html>"

    st.write(" ")
    st.download_button(
        label="🖨️ Imprimir / Guardar Reporte del Turno (Layout Web)",
        data=html_print,
        file_name=f"Plan_Turno_{datetime.now().strftime('%d%m%Y')}.html",
        mime="text/html"
    )

# --- CONFIGURACIONES AL FONDO DE LA PÁGINA ---
st.write("---")
st.header("⚙️ 2. Panel de Ajuste y Configuración de Planta")

m_seleccionadas = st.multiselect(
    "Modifique las máquinas activas en producción:",
    options=list(WORKLOAD_MAESTRO.keys()),
    default=st.session_state.maquinas_activas
)

# Detectar si cambiaron las máquinas activas para relanzar la app de forma segura
if m_seleccionadas != st.session_state.maquinas_activas:
    st.session_state.maquinas_activas = m_seleccionadas
    st.rerun()

st.subheader("🔒 Forzar Asignación Manual y Feedback ML")
if st.session_state.maquinas_activas:
    col_tab = st.columns(3)
    for idx, m in enumerate(sorted(st.session_state.maquinas_activas)):
        with col_tab[idx % 3]:
            with st.expander(f"⚙️ Parámetros Máquina {m}", expanded=False):
                
                # Gestión de prioridad interactiva
                prio_txt = st.selectbox("Prioridad:", ["Media", "Alta", "Baja"], key=f"p_sel_{m}", index=1)
                prio_map = {"Alta": 1, "Media": 2, "Baja": 3}
                st.session_state.prioridades[m] = prio_map[prio_txt]
                
                # Asignación manual de operario fijo
                op_m = st.text_input("Fijar a operario:", value="", key=f"m_input_{m}", placeholder="Ej: Operario 1")
                if op_m.strip():
                    st.session_state.asignaciones_manuales[m] = op_m.strip()
                    mot = st.radio("Motivo del cambio:", ["Condiciones de proceso", "Error distancia", "Baja saturacion", "Error coherencia"], key=f"mot_radio_{m}")
                    registrar_evento_ml(m, mot, op_m.strip())
                elif m in st.session_state.asignaciones_manuales:
                    # Limpiar si el usuario borra el campo de texto
                    del st.session_state.asignaciones_manuales[m]
