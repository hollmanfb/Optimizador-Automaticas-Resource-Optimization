import streamlit as st
import numpy as np

# -------------------------------------------------------------------------
# CONSTANTES Y BASE DE DATOS MAESTRA
# -------------------------------------------------------------------------
MAX_SATURACION = 0.97

WORKLOAD_MAESTRO = {
    "902": 0.3712,
    "903": 0.3437,
    "904": 0.3016,
    "905": 0.3218,
    "906": 0.3289,
    "907": 0.3217,
    "911": 0.1821,
    "916": 0.3868,
    "917": 1.0000,
    "922": 0.5321,
    "923": 0.6995,
    "924": 1.0000,
    "925": 0.3356,
    "926": 0.3361,
    "927": 0.5300,
    "928": 0.6735
}

LAYOUT_COORDENADAS = {
    "922": (1.0, 4.0),
    "907": (3.0, 4.0),
    "902": (4.5, 4.0),
    "927": (6.5, 4.0),
    "904": (9.0, 4.0),
    "916": (9.0, 3.5),
    "911": (1.0, 2.0),
    "926": (1.0, 1.0),
    "925": (2.0, 1.0),
    "905": (3.0, 1.0),
    "903": (4.0, 1.0),
    "924": (6.0, 1.5),
    "923": (9.0, 1.0)
}


def calcular_distancia(m1, m2):
    if m1 in LAYOUT_COORDENADAS and m2 in LAYOUT_COORDENADAS:
        p1 = LAYOUT_COORDENADAS[m1]
        p2 = LAYOUT_COORDENADAS[m2]
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return 5.0


# -------------------------------------------------------------------------
# ALGORITMO DE OPTIMIZACIÓN
# -------------------------------------------------------------------------
def optimizar_asignacion(maquinas_activas, asignaciones_manuales, prioridades):
    maquinas_por_asignar = [
        m for m in maquinas_activas if m not in asignaciones_manuales
    ]

    maquinas_por_asignar.sort(
        key=lambda x: (prioridades.get(x, 1), -WORKLOAD_MAESTRO.get(x, 0))
    )

    operarios = {}

    for maq, op_id in asignaciones_manuales.items():
        if op_id not in operarios:
            operarios[op_id] = {"maquinas": [], "carga_total": 0.0}
        operarios[op_id]["maquinas"].append(maq)
        operarios[op_id]["carga_total"] += WORKLOAD_MAESTRO.get(maq, 0)

    nuevo_op_idx = 1
    while f"Operario {nuevo_op_idx}" in operarios:
        nuevo_op_idx += 1

    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        if op_actual not in operarios:
            operarios[op_actual] = {"maquinas": [], "carga_total": 0.0}

        maquina_pivote = maquinas_por_asignar.pop(0)
        carga_pivote = WORKLOAD_MAESTRO.get(maquina_pivote, 0)

        if operarios[op_actual]["carga_total"] + carga_pivote > MAX_SATURACION:
            if len(operarios[op_actual]["maquinas"]) > 0:
                nuevo_op_idx += 1
                op_actual = f"Operario {nuevo_op_idx}"
                operarios[op_actual] = {
                    "maquinas": [maquina_pivote],
                    "carga_total": carga_pivote
                }
            else:
                operarios[op_actual]["maquinas"].append(maquina_pivote)
                operarios[op_actual]["carga_total"] += carga_pivote
                nuevo_op_idx += 1
            continue

        operarios[op_actual]["maquinas"].append(maquina_pivote)
        operarios[op_actual]["carga_total"] += carga_pivote

        while True:
            candidatas = []
            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO.get(m, 0)
                limite_valido = (
                    operarios[op_actual]["carga_total"] + carga_m
                    <= MAX_SATURACION
                )
                if limite_valido:
                    dist_promedio = np.mean([
                        calcular_distancia(m, ya_asig)
                        for ya_asig in operarios[op_actual]["maquinas"]
                    ])
                    candidatas.append((m, dist_promedio, carga_m))

            if not candidatas:
                break

            candidatas.sort(key=lambda x: x[1])
            mejor_maquina, _, mejor_carga = candidatas[0]

            operarios[op_actual]["maquinas"].append(mejor_maquina)
            operarios[op_actual]["carga_total"] += mejor_carga
            maquinas_por_asignar.remove(mejor_maquina)

        nuevo_op_idx += 1

    return operarios


# -------------------------------------------------------------------------
# INTERFAZ GRÁFICA (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Optimización de Planta")
st.title("🏭 Asignación Autónoma de Operarios")
st.markdown("Algoritmo por proximidad con tope de saturación del **97%**")

st.sidebar.header("🛠️ Configuración del Turno")

maquinas_disponibles = list(WORKLOAD_MAESTRO.keys())
predeterminadas_semana = [
    "927", "902", "922", "911", "905", "907", "903", "923", "924"
]

maquinas_activas = st.sidebar.multiselect(
    "Selecciona las Máquinas Activas:",
    options=maquinas_disponibles,
    default=[m for m in predeterminadas_semana if m in maquinas_disponibles]
)

st.sidebar.subheader("🔒 Asignaciones Manuales")
asignaciones_manuales = {}
for m in maquinas_activas:
    op_manual = st.sidebar.text_input(
        f"Fijar {m} a (Ej: Operario 1):",
        value="",
        key=f"man_{m}"
    )
    if op_manual.strip() != "":
        asignaciones_manuales[m] = op_manual.strip()

st.sidebar.subheader("⭐ Prioridades de Máquina")
prioridades = {}
for m in maquinas_activas:
    prio = st.sidebar.selectbox(
        f"Prioridad {m}:",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: ["Alta", "Media", "Baja"][x-1],
        key=f"prio_{m}"
    )
    prioridades[m] = prio

if maquinas_activas:
    resultado = optimizar_asignacion(
        maquinas_activas,
        asignaciones_manuales,
        prioridades
    )

    resultado = {
        k: v for k, v in resultado.items() if len(v["maquinas"]) > 0
    }
    num_operarios = len(resultado)

    col1, col2, col3 = st.columns(3)
    col1.metric(label="👤 Operarios Mínimos", value=num_operarios)
    col2.metric(label="🏭 Máquinas Activas", value=len(maquinas_activas))
    carga_media = np.mean([
        v["carga_total"] for v in resultado.values()
    ]) * 100
    col3.metric(label="📊 Saturación Media", value=f"{carga_media:.2f}%")

    st.write("---")
    st.subheader("📋 Plan General de Distribución")

    cols_UI = st.columns(min(num_operarios, 4))
    for idx, (operario, datos) in enumerate(resultado.items()):
        with cols_UI[idx % 4]:
            st.info(f"### {operario}")
            st.markdown(f"**Saturación:** `{datos['carga_total'] * 100:.2f}%`")
            st.markdown("**Carga:**")
            for m in datos["maquinas"]:
                es_manual = m in asignaciones_manuales
                tipo_asig = "🔒 Manual" if es_manual else "🤖 Auto"
                porcentaje_carga = WORKLOAD_MAESTRO[m] * 100
                st.write(f"- **Mág. {m}** ({porcentaje_carga:.1f}%) - {tipo_asig}")
else:
    st.warning("⚠️ Selecciona máquinas en el panel izquierdo.")
