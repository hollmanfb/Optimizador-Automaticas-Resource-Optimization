import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MONTAJE AUTOMÁTICO Y DISTANCIAS)
# -------------------------------------------------------------------------
MAX_SATURACION = 0.97  
META_SATURACION = 0.90
MEMORIA_ML = "memoria_aprendizaje.csv"

# Datos adaptados al área de Montaje Automático
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

def registrar_evento_ml(maquina, motivo, operario):
    nuevo = pd.DataFrame([{"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "maquina": maquina, "motivo": motivo, "operario": operario}])
    nuevo.to_csv(MEMORIA_ML, mode='a', header=not os.path.exists(MEMORIA_ML), index=False)

# -------------------------------------------------------------------------
# 2. ALGORITMO BASE CON TOPE ESTRICTO DEL 97%
# -------------------------------------------------------------------------
def optimizar_asignacion(maquinas_activas):
    operarios = {}
    maquinas_por_asignar = []

    # Excepción inicial: Cargas completas (100%) reciben atención dedicada
    for m in maquinas_activas:
        carga_m = WORKLOAD_MAESTRO.get(m, 0)
        if carga_m >= 1.00:
            operarios[f"Operario Dedicado {m}"] = {"maquinas": [m], "carga_total": carga_m}
        else:
            maquinas_por_asignar.append(m)

    maquinas_por_asignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))
    nuevo_op_idx = 1

    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        operarios[op_actual] = {"maquinas": [], "carga_total": 0.0}
        
        pivote = maquinas_por_asignar.pop(0)
        operarios[op_actual]["maquinas"].append(pivote)
        operarios[op_actual]["carga_total"] += WORKLOAD_MAESTRO[pivote]

        while len(maquinas_por_asignar) > 0:
            candidatas = []
            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO[m]
                if operarios[op_actual]["carga_total"] + carga_m <= MAX_SATURACION:
                    dist_base = np.mean([MATRIZ_DISTANCIAS.get(m, {}).get(y, 15.0) for y in operarios[op_actual]["maquinas"]])
                    candidatas.append((m, dist_base))

            if not candidatas: break
            candidatas.sort(key=lambda x: x[1])
            mejor_m = candidatas[0][0]
            
            operarios[op_actual]["maquinas"].append(mejor_m)
            operarios[op_actual]["carga_total"] += WORKLOAD_MAESTRO[mejor_m]
            maquinas_por_asignar.remove(mejor_m)
        nuevo_op_idx += 1

    return operarios

# -------------------------------------------------------------------------
# 3. CONTROL DE ESTADO (ESTABLE DE SESIÓN)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador medmix - Montaje Automático")

if "maquinas_activas" not in st.session_state:
    st.session_state.maquinas_activas = ["927", "902", "922", "911", "905", "907", "903", "923", "924"]

# Inicializar o recalcular la propuesta base de la IA
if "propuesta_actual" not in st.session_state or st. some_trigger_condition if False else True:
    base_ia = optimizar_asignacion(st.session_state.maquinas_activas)
    # Filtrar vacíos
    st.session_state.propuesta_actual = {k: v for k, v in base_ia.items() if len(v["maquinas"]) > 0}

st.title("🏭 Balanceo Dinámico de Cargas - Área de Montaje Automático")
st.markdown("---")

# -------------------------------------------------------------------------
# 4. RENDERIZADO DEL LAYOUT: PROPUESTA INTERACTIVA EN TARJETAS
# -------------------------------------------------------------------------
st.subheader("🚀 1. Plan del Turno Activo (Modifique Máquinas Directamente en el Operario)")

resultado_render = {}
todas_las_maquinas_en_uso = []

# Calcular estados actuales basados en las interacciones de las tarjetas
cols_res = st.columns(min(len(st.session_state.propuesta_actual), 4))

for idx, (operario, datos) in enumerate(sorted(st.session_state.propuesta_actual.items())):
    with cols_res[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            
            # Selector directo para agregar o quitar máquinas asignadas a este operario específico
            maquinas_operario = st.multiselect(
                "Máquinas asignadas:",
                options=st.session_state.maquinas_activas,
