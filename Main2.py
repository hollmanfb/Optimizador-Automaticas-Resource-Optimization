import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MONTAJE AUTOMÁTICO Y DISTANCIAS)
# -------------------------------------------------------------------------
MAX_SATURACION = 0.97  

WORKLOAD_MAESTRO = {
    "902": 0.3712, "903": 0.3437, "904": 0.3016, "905": 0.3218,
    "906": 0.3289, "907": 0.3217, "911": 0.1821, "916": 0.3868,
    "917": 1.0000, "922": 0.5321, "923": 0.6995, "924": 1.0000,
    "925": 0.3356, "926": 0.3361, "927": 0.5300, "928": 0.6735
}

# -------------------------------------------------------------------------
# 2. ALGORITMO DE DISTRIBUCIÓN BASE DE LA IA
# -------------------------------------------------------------------------
def optimizar_asignacion(maquinas_activas):
    operarios = {}
    maquinas_por_asignar = []

    # Identificar máquinas con carga completa (Dedicadas)
    for m in maquinas_activas:
        carga_m = WORKLOAD_MAESTRO.get(m, 0)
        if carga_m >= 1.00:
            operarios[f"Operario Dedicado {m}"] = {"maquinas": [m]}
        else:
            maquinas_por_asignar.append(m)

    maquinas_por_asignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))
    nuevo_op_idx = 1

    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        operarios[op_actual] = {"maquinas": []}
        
        pivote = maquinas_por_asignar.pop(0)
        operarios[op_actual]["maquinas"].append(pivote)

        while len(maquinas_por_asignar) > 0:
            candidatas = []
            carga_actual = sum([WORKLOAD_MAESTRO[x] for x in operarios[op_actual]["maquinas"]])
            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO[m]
                if carga_actual + carga_m <= MAX_SATURACION:
                    candidatas.append(m)

            if not candidatas: break
            mejor_m = candidatas[0]
            operarios[op_actual]["maquinas"].append(mejor_m)
            maquinas_por_asignar.remove(mejor_m)
        nuevo_op_idx += 1

    return operarios

# -------------------------------------------------------------------------
# 3. CONTROL DE ESTADO DE SESIÓN
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador medmix")

if "maquinas_activas" not in st.session_state:
    st.session_state.maquinas_activas = ["927", "902", "922", "911", "905", "907", "903", "923", "924", "928"]

if "propuesta_actual" not in st.session_state:
    base_ia = optimizar_asignacion(st.session_state.maquinas_activas)
    st.session_state.propuesta_actual = {k: v["maquinas"] for k, v in base_ia.items() if len(v["maquinas"]) > 0}

if "prioridades_estrellas" not in st.session_state:
    st.session_state.prioridades_estrellas = {m: "⭐⭐ Media" for m in WORKLOAD_MAESTRO.keys()}

st.title("🏭 Balanceo Dinámico - Área de Montaje Automático")
st.markdown("---")

# -------------------------------------------------------------------------
# 4. GESTIÓN DE EXCLUSIÓN MUTUA DE MÁQUINAS (EVITA DUPLICADOS)
# -------------------------------------------------------------------------
# Reofertamos la lista basándonos en lo que el usuario mueve en la pantalla en tiempo real
maquinas_ocupadas_en_otros = []
for op, maqs in st.session_state.propuesta_actual.items():
    # Recolectamos lo que tiene asignado cada operario actualmente para cruzar datos
    pass

# -------------------------------------------------------------------------
# 5. RENDERIZADO DE LAS TARJETAS DE OPERARIOS
# -------------------------------------------------------------------------
st.subheader("👤 Planificación del Personal y Distribución de Cargas")

cols_res = st.columns(min(len(st.session_state.propuesta_actual), 4))
resultado_final_impresion = {}

for idx, operario in enumerate(sorted(st.session_state.propuesta_actual.keys())):
    maquinas_del_operario = st.session_state.propuesta_actual[operario]
    
    # Calcular qué máquinas están libres para ser elegidas por este operario
    todas_las_asignadas = []
    for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
        if op_ref != operario:
            todas_las_asignadas.extend(maqs_ref)
            
    # Opciones disponibles = Máquinas del catálogo que nadie más tenga asignadas en este momento
    opciones_disponibles = sorted(list(set(st.session_state.maquinas_activas) - set(todas_las_asignadas)))

    with cols_res[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            
            # Selector dinámico: ya no aparecen las máquinas de otros operarios
            nuevas_maquinas = st.multiselect(
                "Máquinas asignadas a este puesto:",
                options=opciones_disponibles,
                default=[m for m in maquinas_del_operario if m in opciones_disponibles],
                key=f"ms_dinamico_{operario}"
            )
            
            # Actualizar el estado global de forma inmediata
            st.session_state.propuesta_actual[operario] = nuevas_maquinas
            
            # Cálculo de saturación real
            carga_real = sum([WORKLOAD_MAESTRO.get(m, 0) for m in nuevas_maquinas])
            sat_p = carga_real * 100
            
            if sat_p > 97.0 and "Dedicado" not in operario:
                st.error(f"🔥 Sobrecarga: {sat_p:.1f}% (Máx 97%)")
            elif "Dedicado" in operario:
                st.warning(f"⚡ Operario Dedicado: {sat_p:.1f}%")
            else:
                st.info(f"⚡ Carga total actual: {sat_p:.1f}%")
            
            # Desplegar prioridades por estrellas dentro de la tarjeta
            if nuevas_maquinas:
                st.write("**Criticidad / Prioridad:**")
                for m in nuevas_maquinas:
                    col_m, col_prio = st.columns([1, 2])
                    with col_m:
                        st.caption(f"🤖 **M-{m}** ({WORKLOAD_MAESTRO[m]*100:.1f}%)")
                    with col_prio:
                        # Selector de estrellas integrado directamente
                        prio_estrella = st.selectbox(
                            f"Prioridad {m}",
                            options=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"],
                            index=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"].index(st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")),
                            label_visibility="collapsed",
                            key=f"prio_star_{operario}_{m}"
                        )
                        st.session_state.prioridades_estrellas[m] = prio_estrella
            else:
                st.caption("⚠️ Puesto sin máquinas asignadas.")
                
            resultado_final_impresion[operario] = {"maquinas": nuevas_maquinas, "carga": sat_p}

# Verificar si alguna máquina activa se quedó huérfana
todas_las_maquinas_en_uso = []
for m_list in st.session_state.propuesta_actual.values():
    todas_las_maquinas_en_uso.extend(m_list)
maquinas_faltantes = set(st.session_state.maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"⚠️ **Atención:** Las siguientes Máquinas Automáticas están activas pero **nadie** las está atendiendo: {', '.join(sorted(maquinas_faltantes))}")

# -------------------------------------------------------------------------
# 6. REPORTE DE IMPRESIÓN ADAPTADO CON ESTRELLAS Y COLORES CORPORATIVOS
# -------------------------------------------------------------------------
st.write("---")
html_print = """
<html>
<head>
<style>
    body { font-family: Arial, sans-serif; color: #222; margin: 20px; }
    .header { border-bottom: 4px solid #1c6e7d; padding-bottom: 12px; margin-bottom: 25px; }
    .title { font-size: 22pt; font-weight: bold; color: #1c6e7d; }
    .subtitle { font-size: 11pt; color: #555; }
    .card { border: 1px solid #bce1e6; border-radius: 6px; margin-bottom: 18px; background: #f4fafb; overflow: hidden; }
    .card-h { background: #1c6e7d; color: white; padding: 12px; font-weight: bold; font-size: 13pt; }
    .card-b { padding: 15px; }
    .badge { background: #d9534f; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10pt; float: right; }
    .badge-normal { background: #2b93a5; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10pt; float: right; }
    ul { margin: 5px 0; padding-left: 20px; }
    li { margin-bottom: 5px; font-size: 11pt; }
</style>
</head>
<body>
    <div class='header'>
        <div class='title'>medmix - HOJA DE RUTA DE MONTAJE AUTOMÁTICO</div>
        <div class='subtitle'>Distribución Dinámica de Operarios y Criticidad de Máquinas</div>
    </div>
"""

for operario, datos in sorted(resultado_final_impresion.items()):
    if datos["maquinas"]:
        clase_badge = "badge" if (datos["carga"] > 97.0 and "Dedicado" not in operario) else "badge-normal"
        html_print += f"""
        <div class='card'>
            <div class='card-h'>{operario} <span class='{clase_badge}'>{datos["carga"]:.1f}% Carga</span></div>
            <div class='card-b'>
                <strong>Plan de trabajo en planta:</strong>
                <ul>
        """
        for m in datos["maquinas"]:
            prio_act = st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")
            html_print += f"<li><strong>Máquina {m}</strong> — Carga: {WORKLOAD_MAESTRO[m]*100:.1f}% | Prioridad: {prio_act}</li>"
        html_print += "</ul></div></div>"

html_print += "</body></html>"

# Botones de acción rápida para reajustar o descargar el reporte
c_b1, c_b2 = st.columns([1, 3])
with c_b1:
    if st.button("🔄 Reiniciar a Propuesta de la IA"):
        st.session_state.propuesta_actual = {k: v["maquinas"] for k, v in optimizar_asignacion(st.session_state.maquinas_activas).items() if len(v["maquinas"]) > 0}
        st.rerun()

with c_b2:
    st.download_button(
        label="🖨️ Exportar e Imprimir Plan de Turno Oficial (Estilo medmix)",
        data=html_print,
        file_name=f"Plan_Montaje_Medmix_{datetime.now().strftime('%d%m%Y')}.html",
        mime="text/html"
    )
