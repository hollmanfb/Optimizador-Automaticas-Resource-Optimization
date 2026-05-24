import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------------
# 1. BASE DE DATOS MAESTRA (MONTAJE AUTOMÁTICO, CARGAS Y DISTANCIAS REALES)
# -------------------------------------------------------------------------
MAX_SATURACION = 0.97  

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

# -------------------------------------------------------------------------
# 2. MOTOR DE OPTIMIZACIÓN BASADO EN PROXIMIDAD GEOGRÁFICA (METROS)
# -------------------------------------------------------------------------
def optimizar_con_distancias(maquinas_trabajando):
    operarios = {}
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    # 1. Separar máquinas dedicadas (Carga >= 100%)
    for m in list(maquinas_por_asignar):
        if WORKLOAD_MAESTRO.get(m, 0) >= 1.00:
            operarios[f"Operario Dedicado {m}"] = {"maquinas": [m]}
            maquinas_por_asignar.remove(m)

    # 2. Agrupar el resto por cercanía estricta en metros
    maquinas_por_asignar.sort(key=lambda x: -WORKLOAD_MAESTRO.get(x, 0))
    nuevo_op_idx = 1

    while len(maquinas_por_asignar) > 0:
        op_actual = f"Operario {nuevo_op_idx}"
        pivote = maquinas_por_asignar.pop(0)
        operarios[op_actual] = {"maquinas": [pivote]}
        
        while len(maquinas_por_asignar) > 0:
            carga_actual = sum([WORKLOAD_MAESTRO[x] for x in operarios[op_actual]["maquinas"]])
            candidatas_cercanas = []

            for m in maquinas_por_asignar:
                carga_m = WORKLOAD_MAESTRO[m]
                if carga_actual + carga_m <= MAX_SATURACION:
                    # Distancia promedio de la nueva máquina a las ya asignadas en esta tarjeta
                    dist_promedio = np.mean([MATRIZ_DISTANCIAS[m].get(ya_asignada, 50.0) for ya_asignada in operarios[op_actual]["maquinas"]])
                    candidatas_cercanas.append((m, dist_promedio))
            
            if not candidatas_cercanas:
                break
            
            # Ordenar por menor distancia en metros (Evita emparejar 911 con 923)
            candidatas_cercanas.sort(key=lambda x: x[1])
            mejor_maquina = candidatas_cercanas[0][0]
            
            operarios[op_actual]["maquinas"].append(mejor_maquina)
            maquinas_por_asignar.remove(mejor_maquina)
            
        nuevo_op_idx += 1

    return operarios

# -------------------------------------------------------------------------
# 3. INTERFAZ Y CONTROL DE ESTADOS (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador medmix")

# Inicialización de estados de sesión
if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in WORKLOAD_MAESTRO.keys()}
    # Ajuste inicial por defecto de algunas en Día Libre si se requiere
    for desactiva in ["904", "906", "916", "917", "925", "926", "928"]:
        st.session_state.estados_maquinas[desactiva] = "Día Libre"

if "prioridades_estrellas" not in st.session_state:
    st.session_state.prioridades_estrellas = {m: "⭐⭐ Media" for m in WORKLOAD_MAESTRO.keys()}

# --- COLUMNA IZQUIERDA (CONTROL DE ESTADOS DISPONIBLES) ---
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    st.header("⚙️ Estado de las Máquinas")
    st.write("Cambia a 'Día Libre' para retirar una máquina del turno:")
    
    estados_actualizados = {}
    for m in sorted(WORKLOAD_MAESTRO.keys()):
        estado_previo = st.session_state.estados_maquinas.get(m, "Trabajando")
        opcion = st.selectbox(
            f"Máquina {m} (Carga: {WORKLOAD_MAESTRO[m]*100:.1f}%)",
            options=["Trabajando", "Día Libre"],
            index=0 if estado_previo == "Trabajando" else 1,
            key=f"sb_estado_{m}"
        )
        estados_actualizados[m] = opcion

    # Detectar cambios en los interruptores laterales para recalcular de inmediato
    if estados_actualizados != st.session_state.estados_maquinas:
        st.session_state.estados_maquinas = estados_actualizados
        maquinas_activas = [k for k, v in estados_actualizados.items() if v == "Trabajando"]
        base_ia = optimizar_con_distancias(maquinas_activas)
        st.session_state.propuesta_actual = {k: v["maquinas"] for k, v in base_ia.items()}
        st.rerun()

# Filtrar lista de ejecución en base a la columna izquierda
maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]

if "propuesta_actual" not in st.session_state:
    base_ia = optimizar_con_distancias(maquinas_activas)
    st.session_state.propuesta_actual = {k: v["maquinas"] for k, v in base_ia.items()}

st.title("🏭 Balanceo de Línea con Matriz de Distancia Física")
st.caption("Área de Montaje Automático — Respetando distancias de celdas y topes de saturación.")
st.markdown("---")

# -------------------------------------------------------------------------
# 4. ENTORNO DE ASIGNACIÓN INTERACTIVA EN TARJETAS
# -------------------------------------------------------------------------
st.subheader("🚀 1. Control de Operarios y Asignación Manual")

resultado_final_impresion = {}
cols_res = st.columns(min(max(len(st.session_state.propuesta_actual), 1), 4))

for idx, operario in enumerate(sorted(list(st.session_state.propuesta_actual.keys()))):
    maquinas_del_operario = st.session_state.propuesta_actual.get(operario, [])
    
    # Exclusión Mutua en tiempo real: remover asignadas a otros del desplegable
    otras_asignadas = []
    for op_ref, maqs_ref in st.session_state.propuesta_actual.items():
        if op_ref != operario:
            otras_asignadas.extend(maqs_ref)
            
    opciones_disponibles = sorted(list(set(maquinas_activas) - set(otras_asignadas)))

    with cols_res[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            
            nuevas_maquinas = st.multiselect(
                "Añadir/Quitar Máquinas:",
                options=opciones_disponibles,
                default=[m for m in maquinas_del_operario if m in opciones_disponibles],
                key=f"ms_tarjeta_{operario}"
            )
            
            # Guardar selección manual en la sesión
            st.session_state.propuesta_actual[operario] = nuevas_maquinas
            
            # Validación de métricas (Carga + Distancias internas en la tarjeta)
            carga_real = sum([WORKLOAD_MAESTRO.get(m, 0) for m in nuevas_maquinas])
            sat_p = carga_real * 100
            
            # Mostrar distancias internas críticas si tiene más de una máquina
            alerta_distancia = False
            distancias_texto = []
            if len(nuevas_maquinas) > 1:
                for i in range(len(nuevas_maquinas)):
                    for j in range(i + 1, len(nuevas_maquinas)):
                        m1, m2 = nuevas_maquinas[i], nuevas_maquinas[j]
                        dist = MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 0)
                        distancias_texto.append(f"Distancia {m1}↔️{m2}: {dist}m")
                        if dist > 20: # Alerta si supera los 20 metros de traslado
                            alerta_distancia = True

            # Colores informativos de carga
            if sat_p > 97.0 and "Dedicado" not in operario:
                st.error(f"🔥 Sobrecarga: {sat_p:.1f}% (Máx 97%)")
            elif "Dedicado" in operario:
                st.warning(f"⚡ Operario Dedicado: {sat_p:.1f}%")
            else:
                st.info(f"⚡ Carga total: {sat_p:.1f}%")

            # Desplegar avisos de distancias largas dentro de la tarjeta
            if distancias_texto:
                with st.expander("📍 Ver trayectos en metros", expanded=alerta_distancia):
                    for txt in distancias_texto:
                        if "37m" in txt or "40m" in txt or "44m" in txt:
                            st.write(f"❌ {txt} — **¡Muy lejos!**")
                        else:
                            st.write(f"✅ {txt}")

            # Sistema de Prioridades por Estrellas (⭐)
            if nuevas_maquinas:
                st.write("**Criticidad / Estrellas:**")
                for m in nuevas_maquinas:
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.caption(f"🤖 **M-{m}**")
                    with c2:
                        prio_estrella = st.selectbox(
                            f"Prio_{operario}_{m}",
                            options=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"],
                            index=["⭐⭐⭐ Alta", "⭐⭐ Media", "⭐ Baja"].index(st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")),
                            label_visibility="collapsed",
                            key=f"star_sel_{operario}_{m}"
                        )
                        st.session_state.prioridades_estrellas[m] = prio_estrella
                        
            resultado_final_impresion[operario] = {"maquinas": nuevas_maquinas, "carga": sat_p}

# Alertas globales de máquinas desatendidas
todas_las_maquinas_en_uso = []
for m_list in st.session_state.propuesta_actual.values():
    todas_las_maquinas_en_uso.extend(m_list)
maquinas_faltantes = set(maquinas_activas) - set(todas_las_maquinas_en_uso)

if maquinas_faltantes:
    st.error(f"⚠️ **Máquinas sin cubrir:** Tienes máquinas en estado 'Trabajando' que no han sido asignadas: {', '.join(sorted(maquinas_faltantes))}")

# -------------------------------------------------------------------------
# 5. REPORTE DE IMPRESIÓN ADAPTADO (ESTILO MEDMIX CON PRIORIDADES Y METROS)
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
    .badge-normal { background: #2b93a5; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10pt; float: right; }
    ul { margin: 5px 0; padding-left: 20px; }
    li { margin-bottom: 5px; font-size: 11pt; }
</style>
</head>
<body>
    <div class='header'>
        <div class='title'>medmix - HOJA DE TRABAJO DE MONTAJE AUTOMÁTICO</div>
        <div class='subtitle'>Reporte Verde de Distribución de Cargas y Matriz de Proximidad</div>
    </div>
"""

for operario, datos in sorted(resultado_final_impresion.items()):
    if datos["maquinas"]:
        html_print += f"""
        <div class='card'>
            <div class='card-h'>{operario} <span class='badge-normal'>{datos["carga"]:.1f}% Carga</span></div>
            <div class='card-b'>
                <strong>Hoja de Ruta del Puesto:</strong>
                <ul>
        """
        for m in datos["maquinas"]:
            prio_act = st.session_state.prioridades_estrellas.get(m, "⭐⭐ Media")
            html_print += f"<li><strong>Máquina {m}</strong> — Carga: {WORKLOAD_MAESTRO[m]*100:.1f}% | Prioridad: {prio_act}</li>"
        html_print += "</ul></div></div>"

html_print += "</body></html>"

col_acc1, col_acc2 = st.columns([1, 3])
with col_acc1:
    if st.button("🔄 Recalcular por Proximidad (IA)"):
        base_ia = optimizar_con_distancias(maquinas_activas)
        st.session_state.propuesta_actual = {k: v["maquinas"] for k, v in base_ia.items()}
        st.rerun()

with col_acc2:
    st.download_button(
        label="🖨️ Descargar Documento para Impresión de Planta",
        data=html_print,
        file_name=f"Ruta_Montaje_Medmix_{datetime.now().strftime('%d%m%Y')}.html",
        mime="text/html"
    )
