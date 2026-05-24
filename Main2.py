import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ARCHIVOS DE MEMORIA (MACHINE LEARNING)
# -------------------------------------------------------------------------
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
    "904": {"902":23, "903":27, "904":0, "905":39, "906":26, "907":29, "911":44, "916":3, "917":23, "922":43, "923":13, "924":24, "925":42, "926":45, "927":15, "928":21},
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
    """Carga los errores del histórico para penalizar distancias (Aprendizaje)."""
    penalizaciones = {}
    if os.path.exists(MEMORIA_ML):
        try:
            df = pd.read_csv(MEMORIA_ML)
            # Solo penalizar si fue por error de distancia o coherencia
            df_errores = df[df["motivo"].str.contains("distancias|coherencia", case=False, na=False)]
            for _, fila in df_errores.iterrows():
                m = str(fila["maquina"])
                if m not in penalizaciones:
                    penalizaciones[m] = 50.0  # Añade una barrera virtual de 50 metros artificiales
        except Exception:
            pass
    return penalizaciones

def registrar_evento_ml(maquina, motivo, operario):
    """Guarda en caliente el motivo de la asignación manual para entrenar al modelo."""
    nuevo_registro = pd.DataFrame([{
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "maquina": maquina,
        "motivo": motivo,
        "operario_forzado": operario
    }])
    if not os.path.exists(MEMORIA_ML):
        nuevo_registro.to_csv(MEMORIA_ML, index=False)
    else:
        nuevo_registro.to_csv(MEMORIA_ML, mode='a', header=False, index=False)

def guardar_historico_impresion(plan_texto):
    """Guarda el plan definitivo en el historial de la planta al pulsar imprimir."""
    nuevo_plan = pd.DataFrame([{
        "fecha_impresion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detalle_plan": plan_texto.replace('\n', ' | ')
    }])
    if not os.path.exists(HISTORICO_ASIGNACIONES):
        nuevo_plan.to_csv(HISTORICO_ASIGNACIONES, index=False)
    else:
        nuevo_plan.to_csv(HISTORICO_ASIGNACIONES, mode='a', header=False, index=False)

# -------------------------------------------------------------------------
# 2. ALGORITMO CON APRENDIZAJE POR RETROALIMENTACIÓN
# -------------------------------------------------------------------------
def optimizar_asignacion(maquinas_activas, asignaciones_manuales, prioridades):
    operarios = {}
    maquinas_por_asignar = []
    
    # Cargar el Machine Learning (Barreras lógicas aprendidas de turnos pasados)
    barreras_ml = cargar_penalizaciones_ml()

    # Fase 1: Asignaciones forzadas y máquinas críticas al 100%
    for m in maquinas_activas:
        carga_m = WORKLOAD_MAESTRO.get(m, 0)
        if m in asignaciones_manuales:
            op_id = asignaciones_manuales[m]
            if op_id not in operarios:
                operarios[op_id] = {"maquinas": [], "carga_total": 0.0}
            operarios[op_id]["maquinas"].append(m)
            operarios[op_id]["carga_total"] += carga_m
        elif carga_m >= 1.00:
            op_dedicado = f"Operario Dedicado {m}"
            operarios[op_dedicado] = {"maquinas": [m], "carga_total": carga_m}
        else:
            maquinas_por_asignar.append(m)

    # Ordenar combinando Prioridad de Negocio y volumen de carga
    maquinas_por_asignar.sort(key=lambda x: (prioridades.get(x, 2), -WORKLOAD_MAESTRO.get(x, 0)))

    nuevo_op_idx = 1
    while f"Operario {nuevo_op_idx}" in operarios:
        nuevo_op_idx += 1

    # Fase 2: Reparto dinámico inteligente (Matriz + Penalización Inteligente)
    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        if op_actual not in operarios:
            operarios[op_actual] = {"maquinas": [], "carga_total": 0.0}

        maquina_pivote = maquinas_por_asignar.pop(0)
        operarios[op_actual]["maquinas"].append(maquina_pivote)
        operarios[op_actual]["carga_total"] += WORKLOAD_MAESTRO.get(maquina_pivote, 0)

        while operarios[op_actual]["carga_total"] < META_SATURACION and len(maquinas_por_asignar) > 0:
            candidatas = []
            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO.get(m, 0)
                
                # Calcular distancia base en metros
                dist_base = np.mean([
                    MATRIZ_DISTANCIAS.get(m, {}).get(ya_asig, 15.0)
                    for ya_asig in operarios[op_actual]["maquinas"]
                ])
                
                # APLICACIÓN DEL MACHINE LEARNING: Si esta máquina provocó un error antes,
                # se le añade una penalización artificial de distancia para alejarla del grupo.
                penalizacion_ai = barreras_ml.get(m, 0.0)
                dist_final_ajustada = dist_base + penalizacion_ai
                
                candidatas.append((m, dist_final_ajustada, carga_m))

            candidatas.sort(key=lambda x: x[1])
            mejor_maquina, _, mejor_carga = candidatas[0]

            operarios[op_actual]["maquinas"].append(mejor_maquina)
            operarios[op_actual]["carga_total"] += mejor_carga
            maquinas_por_asignar.remove(mejor_maquina)

        nuevo_op_idx += 1

    return operarios

# -------------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA AVANZADA (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador Inteligente ML")
st.title("🏭 Planificador Autónomo con Auto-Aprendizaje (ML)")
st.markdown("El sistema aprende de tus asignaciones manuales para corregir su lógica de proximidad.")

if "maquinas_activas" not in st.session_state:
    st.session_state.maquinas_activas = ["927", "902", "922", "911", "905", "907", "903", "923", "924"]

st.subheader("🛠️ 1. Estado de la Planta")
st.session_state.maquinas_activas = st.multiselect(
    "Selecciona las máquinas que van a trabajar en este turno:",
    options=list(WORKLOAD_MAESTRO.keys()),
    default=st.session_state.maquinas_activas
)

st.write("---")
st.subheader("📋 2. Ajustes Manuales y Entrenamiento del Machine Learning")

asignaciones_manuales = {}
prioridades = {}

if st.session_state.maquinas_activas:
    columnas_tabla = st.columns(3)
    for idx, m in enumerate(sorted(st.session_state.maquinas_activas)):
        col_seleccionada = columnas_tabla[idx % 3]
        with col_seleccionada:
            carga_p = WORKLOAD_MAESTRO[m] * 100
            with st.expander(f"📦 Máquina {m} ({carga_p:.1f}%)", expanded=True):
                prioridades[m] = st.selectbox("Prioridad:", ["Media", "Alta", "Baja"], key=f"p_{m}")
                prioridades[m] = {"Alta": 1, "Media": 2, "Baja": 3}[prioridades[m]]
                
                op_manual = st.text_input("Asignar a un operario fijo:", value="", key=f"m_{m}", placeholder="Ej: Operario 1")
                
                if op_manual.strip():
                    asignaciones_manuales[m] = op_manual.strip()
                    
                    # DESPLIEGUE INMEDIATO DE MOTIVOS REQUERIDOS POR EL USUARIO
                    motivo = st.radio(
                        f"Motivo del cambio en Máq. {m}:",
                        options=[
                            "Asignacion por condiciones del proceso",
                            "Error de asignacion por distancias",
                            "Error de asignacion baja saturacion",
                            "Error de asignacion por coherencia"
                        ],
                        key=f"motivo_{m}"
                    )
                    # Guardar la retroalimentación en caliente para entrenar el algoritmo
                    registrar_evento_ml(m, motivo, op_manual.strip())

    # Procesar optimización
    st.write("---")
    st.subheader("🚀 3. Distribución Óptima de Turno Propuesta")
    
    resultado = optimizar_asignacion(st.session_state.maquinas_activas, asignaciones_manuales, prioridades)
    resultado = {k: v for k, v in resultado.items() if len(v["maquinas"]) > 0}
    num_operarios = len(resultado)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="👤 Operarios Mínimos", value=num_operarios)
    kpi2.metric(label="🏭 Máquinas Activas", value=len(st.session_state.maquinas_activas))
    carga_media = np.mean([v["carga_total"] for v in resultado.values()]) * 100
    kpi3.metric(label="📊 Saturación Media Obtenida", value=f"{carga_media:.2f}%")

    texto_impresion = f"REPARTO DE TURNO - GENERADO EL {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    texto_impresion += "==============================================================\n"
    
    cols_resultado = st.columns(min(num_operarios, 4))
    for idx, (operario, datos) in enumerate(sorted(resultado.items())):
        sat_porc = datos['carga_total'] * 100
        texto_impresion += f"{operario} ({sat_porc:.1f}% Carga) -> Maquinas: {', '.join(datos['maquinas'])}\n"
        
        with cols_resultado[idx % 4]:
            st.success(f"### {operario}")
            st.metric(label="Carga total", value=f"{sat_porc:.1f}%")
            for m in datos["maquinas"]:
                tipo = "🔒 Manual" if m in asignaciones_manuales else "🤖 IA"
                st.write(f"- **Máq. {m}** ({WORKLOAD_MAESTRO[m]*100:.1f}%) - {tipo}")
                
    st.write("---")
    
    # ACCIÓN COMBINADA: Descarga el archivo de impresión y guarda el log histórico en el servidor
    if st.download_button(
        label="🖨️ Descargar e Imprimir Plan de Trabajo",
        data=texto_impresion,
        file_name="plan_turno_inyectoras.txt",
        mime="text/plain"
    ):
        guardar_historico_impresion(texto_impresion)
        st.toast("✅ ¡Plan guardado con éxito en el histórico de la planta!")

else:
    st.warning("⚠️ Selecciona máquinas en el paso 1.")
