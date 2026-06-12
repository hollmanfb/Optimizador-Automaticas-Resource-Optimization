import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# -------------------------------------------------------------------------
# PARAMETRIZACIÓN BASE DE INGENIERÍA (MEDMIX)
# -------------------------------------------------------------------------
MAX_SATURACION_ESTANDAR = 1.10  
VELOCIDAD_CAMINADO = 1.2  
FRECUENCIA_TRASLADOS_HORA = 4  

WORKLOAD_MAESTRO_BASE = {
    "902": 0.3712, "903": 0.3440, "904": 0.3020, "905": 0.3220,
    "906": 0.3290, "907": 0.3220, "911": 0.1820, "916": 0.3870,
    "917": 1.0000, "922": 0.5320, "923": 0.7000, "924": 1.0000,
    "925": 0.3360, "926": 0.3360, "927": 0.5300, "928": 0.6730
}

MATRIZ_DISTANCIAS = {
    "902": {"902":0, "903":7, "904":23, "905":9, "906":4, "907":8, "911":25, "916":21, "917":41, "922":21, "923":27, "924":15, "925":12, "926":26, "927":6, "928":40},
    "903": {"902":7, "903":0, "904":27, "905":10, "906":8, "907":8.5, "911":20, "916":26, "917":48, "922":21, "923":26, "924":12, "925":15, "926":20, "927":11, "928":47},
    "904": {"902":23, "903":27, "904":0, "905":39, "906":26, "907":29, "911":44, "916":3, "917":27, "922":43, "923":13, "924":24, "925":42, "926":45, "927":15, "928":21},
    "905": {"902":9, "903":10, "904":39, "905":0, "906":12, "907":10, "911":9, "916":28, "917":50, "922":12, "923":28, "924":18, "925":3, "926":6, "927":18, "928":35},
    "906": {"902":4, "903":8, "904":26, "905":12, "906":0, "907":2, "911":20, "916":37, "917":49, "922":19, "923":36, "924":20, "925":18, "926":21, "927":12, "928":37},
    "907": {"902":8, "903":8.5, "904":29, "905":10, "906":2, "907":0, "911":18, "916":25, "917":51, "922":17, "923":38, "924":22, "925":16, "926":19, "927":14, "928":39},
    "911": {"902":25, "903":20, "904":44, "905":9, "906":20, "907":18, "911":0, "916":37, "917":59, "922":1, "923":37, "924":27, "925":3, "926":1, "927":27, "928":44},
    "916": {"902":21, "903":26, "904":3, "905":28, "906":37, "907":25, "911":37, "916":0, "917":27, "922":40, "923":10, "924":21, "925":39, "926":42, "927":12, "928":26},
    "917": {"902":41, "903":48, "904":23, "905":50, "906":49, "907":51, "911":59, "916":27, "917":0, "922":70, "923":38, "924":74, "925":78, "926":80, "927":47, "928":16},
    "922": {"902":21, "903":21, "904":43, "905":12, "906":19, "907":17, "911":1, "916":40, "917":70, "922":0, "923":50, "924":36, "925":10, "926":13, "927":37, "928":62},
    "923": {"902":27, "903":26, "904":13, "905":28, "906":36, "907":38, "911":37, "916":10, "917":38, "922":50, "923":0, "924":19, "925":42, "926":46, "927":22, "928":46},
    "924": {"902":15, "903":12, "904":24, "905":18, "906":20, "907":22, "911":27, "916":21, "917":74, "922":36, "923":19, "924":0, "925":24, "926":28, "927":20, "928":45},
    "925": {"902":12, "903":15, "904":42, "905":3, "906":18, "907":16, "911":3, "916":39, "917":78, "922":10, "923":42, "924":24, "925":0, "926":3, "927":35, "928":62},
    "926": {"902":26, "903":20, "904":45, "905":6, "906":21, "907":19, "911":1, "916":42, "917":80, "922":13, "923":46, "924":28, "925":3, "926":0, "927":38, "928":66},
    "927": {"902":6, "903":11, "904":15, "905":18, "906":12, "907":14, "911":27, "916":12, "917":47, "922":37, "923":22, "924":20, "925":35, "926":38, "927":0, "928":25},
    "928": {"902":40, "903":40, "904":21, "905":35, "906":37, "907":39, "911":44, "916":26, "917":16, "922":62, "923":46, "924":45, "925":62, "926":66, "927":25, "928":0}
}

LISTA_8_OPERARIOS = [f"Operario {i}" for i in range(1, 9)]

def calcular_carga_caminado(lista_maquinas):
    if not lista_maquinas or len(lista_maquinas) <= 1:
        return 0.0
    distancias = []
    for i in range(len(lista_maquinas)):
        for j in range(i + 1, len(lista_maquinas)):
            m1, m2 = lista_maquinas[i], lista_maquinas[j]
            distancias.append(MATRIZ_DISTANCIAS.get(m1, {}).get(m2, 25.0))
    return (np.mean(distancias) / VELOCIDAD_CAMINADO * FRECUENCIA_TRASLADOS_HORA) / 3600.0

# -------------------------------------------------------------------------
# MOTOR DE OPTIMIZACIÓN ABSTRACTO (TRATA A LOS OPERARIOS COMO IRRELEVANTES)
# -------------------------------------------------------------------------
def optimizar_celdas_abstractas(maquinas_trabajando, num_operarios_disponibles, cargas_activas, variante_902):
    """
    Crea bloques de trabajo óptimos independientes de a qué operario se le asigne.
    Retorna una lista de listas (bloques).
    """
    bloques = []
    maquinas_por_asignar = [m for m in maquinas_trabajando]

    if num_operarios_disponibles <= 0:
        return bloques

    # 1. Extraer cargas completas críticas inamovibles (917, 924, 928)
    for m in ["917", "924", "928"]:
        if m in maquinas_por_asignar:
            bloques.append([m])
            maquinas_por_asignar.remove(m)

    # 2. Agrupación estructural natural del Bloque Mecánico (916 + 904)
    bloque_mecanico = []
    for m in ["916", "904"]:
        if m in maquinas_por_asignar:
            bloque_mecanico.append(m)
            maquinas_por_asignar.remove(m)
    if bloque_mecanico:
        bloques.append(bloque_mecanico)

    # 3. Lógica operativa del pasillo central según Variante de Producción
    if variante_902 == "Cánula Larga (45.0%)":
        # Bloque Estratégico Dorado (902 + 906)
        bloque_dorado = []
        for m in ["902", "906"]:
            if m in maquinas_por_asignar:
                bloque_dorado.append(m)
                maquinas_por_asignar.remove(m)
        if bloque_dorado:
            bloques.append(bloque_dorado)
            
        # El resto del pasillo (907, 903) conforma otro grupo natural
        bloque_pasillo = []
        for m in ["907", "903"]:
            if m in maquinas_por_asignar:
                bloque_pasillo.append(m)
                maquinas_por_asignar.remove(m)
        if bloque_pasillo:
            bloques.append(bloque_pasillo)
    else:
        # Cánula Corta: Todo el pasillo central unificado (906 + 907 + 903)
        bloque_central = []
        for m in ["906", "907", "903"]:
            if m in maquinas_por_asignar:
                bloque_central.append(m)
                maquinas_por_asignar.remove(m)
        if bloque_central:
            bloques.append(bloque_central)

    # 4. Bloque Pasillo Izquierdo (922 + 911 + 905)
    bloque_izq = []
    for m in ["922", "911", "905"]:
        if m in maquinas_por_asignar:
            bloque_izq.append(m)
            maquinas_por_asignar.remove(m)
    if bloque_izq:
        bloques.append(bloque_izq)

    # 5. La celda 927 (Fin de línea derecho) busca emparejarse si queda espacio libre
    if "927" in maquinas_por_asignar:
        # En Cánula Corta, si la 902 no fue absorbida por el pasillo y está libre, se une a la 927
        if variante_902 == "Cánula Corta (37.1%)" and "902" in maquinas_por_asignar:
            bloques.append(["927", "902"])
            maquinas_por_asignar.remove("927")
            maquinas_por_asignar.remove("902")
        else:
            bloques.append(["927"])
            maquinas_por_asignar.remove("927")

    # 6. Distribución de Remanentes puros (ej. 923) por balanceo dinámico y cercanía real
    maquinas_por_asignar.sort(key=lambda x: -cargas_activas.get(x, 0))
    for m in list(maquinas_por_assignar):
        mejor_bloque_idx = None
        menor_incremento_caminado = float('inf')
        
        for idx, blk in enumerate(bloques):
            # Pokayoke de distancias prohibitivas dentro de un mismo bloque
            if m == "904" and "928" in blk: continue
            if m == "928" and "904" in blk: continue
            if variante_902 == "Cánula Larga (45.0%)" and "927" in blk and m == "923": continue

            test_blk = blk + [m]
            carga_estatica = sum([cargas_activas[x] for x in test_blk])
            carga_dinamica = calcular_carga_caminado(test_blk)
            
            if (carga_estatica + carga_dinamica) <= MAX_SATURACION_ESTANDAR:
                if carga_dinamica < menor_incremento_caminado:
                    menor_incremento_caminado = carga_dinamica
                    mejor_bloque_idx = idx
                    
        if mejor_bloque_idx is not None:
            bloques[mejor_bloque_idx].append(m)
            maquinas_por_asignar.remove(m)

    # 7. Forzado absoluto final si quedan huérfanas
    for m in list(maquinas_por_asignar):
        if not bloques:
            bloques.append([m])
        else:
            idx_menos_cargado = min(range(len(bloques)), key=lambda i: sum([cargas_activas[x] for x in bloques[i]]) + calcular_carga_caminado(bloques[i]))
            bloques[idx_menos_cargado].append(m)
        maquinas_por_asignar.remove(m)

    # Ajustar cantidad de bloques generados a la cantidad real de operarios disponibles en planta
    while len(bloques) > num_operarios_disponibles:
        blk_a_disolver = bloques.pop()
        for m in blk_a_disolver:
            idx_destino = min(range(len(bloques)), key=lambda i: sum([cargas_activas[x] for x in bloques[i]]) + calcular_carga_caminado(bloques[i]))
            bloques[idx_destino].append(m)

    return bloques

# -------------------------------------------------------------------------
# INTERFAZ Y REFRESCO DE DATOS (STREAMLIT)
# -------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Planificador de Cargas medmix")

if "version_902" not in st.session_state:
    st.session_state.version_902 = "Cánula Corta (37.1%)"
if "sync_version" not in st.session_state:
    st.session_state.sync_version = 0

cargas_dinamicas_turno = WORKLOAD_MAESTRO_BASE.copy()
cargas_dinamicas_turno["902"] = 0.4500 if st.session_state.version_902 == "Cánula Larga (45.0%)" else 0.3712

if "estados_maquinas" not in st.session_state:
    st.session_state.estados_maquinas = {m: "Trabajando" for m in cargas_dinamicas_turno.keys()}
    for inactiva in ["925", "926"]:
        st.session_state.estados_maquinas[inactiva] = "Día Libre"

if "estados_operarios" not in st.session_state:
    st.session_state.estados_operarios = {op: "Disponible" for op in LISTA_8_OPERARIOS}

# Inicialización limpia de mapa de asignación
if "mapa_asignaciones" not in st.session_state:
    st.session_state.mapa_asignaciones = {op: [] for op in LISTA_8_OPERARIOS}

maquinas_activas = [k for k, v in st.session_state.estados_maquinas.items() if v == "Trabajando"]
ops_activos = [k for k, v in st.session_state.estados_operarios.items() if v == "Disponible"]

# -------------------------------------------------------------------------
# PANEL LATERAL (SIDEBAR)
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://www.medmix.mixpac.com/images/medmix_Logo_Pos_RGB.svg", width=180)
    st.markdown("### 🏃 Ergonómico: **1.2 m/s**")
    st.markdown("---")
    
    version_sel = st.selectbox(
        "M-902 - Tipo de Cánula:", 
        options=["Cánula Corta (37.1%)", "Cánula Larga (45.0%)"], 
        index=0 if st.session_state.version_902 == "Cánula Corta (37.1%)" else 1
    )
    if version_sel != st.session_state.version_902:
        st.session_state.version_902 = version_sel
        st.session_state.sync_version += 1
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Estado de Celdas")
    for m in sorted(list(cargas_dinamicas_turno.keys())):
        est_actual = st.session_state.estados_maquinas.get(m, "Trabajando")
        col_lbl, col_btn = st.columns([2, 1])
        with col_lbl:
            st.markdown(f"**{m}** ({cargas_dinamicas_turno[m]*100:.1f}%)")
        with col_btn:
            if est_actual == "Trabajando":
                if st.button("🟢", key=f"p_{m}"):
                    st.session_state.estados_maquinas[m] = "Día Libre"
                    st.session_state.sync_version += 1
                    st.rerun()
            else:
                if st.button("🔴", key=f"a_{m}"):
                    st.session_state.estados_maquinas[m] = "Trabajando"
                    st.session_state.sync_version += 1
                    st.rerun()

    st.markdown("---")
    st.markdown("### 👤 Disponibilidad de Personal")
    for op in LISTA_8_OPERARIOS:
        estado_previo = st.session_state.estados_operarios.get(op, "Disponible")
        sel_op = st.selectbox(f"{op}:", options=["Disponible", "Día Libre"], key=f"s_{op}", index=0 if estado_previo == "Disponible" else 1)
        if sel_op != estado_previo:
            st.session_state.estados_operarios[op] = sel_op
            st.session_state.sync_version += 1
            st.rerun()

# -------------------------------------------------------------------------
# CAPA LOGÍSTICA DE MAPEO (ELIMINA LA DISCRIMINACIÓN ENTRE OPERARIOS)
# -------------------------------------------------------------------------
def aplicar_optimizacion_maestra():
    bloques_optimos = optimizar_celdas_abstractas(maquinas_activas, len(ops_activos), cargas_dinamicas_turno, st.session_state.version_902)
    nuevo_mapa = {op: [] for op in LISTA_8_OPERARIOS}
    for idx, op in enumerate(ops_activos):
        if idx < len(bloques_optimos):
            nuevo_mapa[op] = bloques_optimos[idx]
    st.session_state.mapa_asignaciones = nuevo_mapa

# Si las asignaciones actuales están vacías y hay máquinas, inicializar de forma limpia
maquinas_mapeadas = []
for op in ops_activos:
    maquinas_mapeadas.extend(st.session_state.mapa_asignaciones.get(op, []))
if not maquinas_mapeadas and maquinas_activas:
    aplicar_optimizacion_maestra()

# -------------------------------------------------------------------------
# INTERFAZ CENTRAL: TARJETAS 100% SIMÉTRICAS E INDEPENDIENTES
# -------------------------------------------------------------------------
st.markdown("## 📊 Distribución Dinámica de Células de Trabajo")

# Control de Cobertura en tiempo real
maquinas_en_pantalla = []
for op in ops_activos:
    maquinas_en_pantalla.extend(st.session_state.mapa_asignaciones.get(op, []))
maquinas_huerfanas = sorted(list(set(maquinas_activas) - set(maquinas_en_pantalla)))

if maquinas_huerfanas:
    st.error(f"⚠️ ATENCIÓN: Hay celdas operando sin ningún operario asignado: {', '.join(maquinas_huerfanas)}")
else:
    st.success("✅ Cobertura Total: Todas las celdas asignadas eficientemente.")

# Renderizado de Tarjetas (Absolutamente idénticas para todos)
cols_tarjetas = st.columns(4)
for idx, operario in enumerate(LISTA_8_OPERARIOS):
    esta_disponible = st.session_state.estados_operarios.get(operario, "Disponible") == "Disponible"
    
    with cols_tarjetas[idx % 4]:
        with st.container(border=True):
            st.markdown(f"### 👤 {operario}")
            
            if not esta_disponible:
                st.markdown("<span style='color:grey; font-style:italic;'>❌ Turno Libre / Ausencia</span>", unsafe_allow_html=True)
                st.session_state.mapa_asignaciones[operario] = []
            else:
                # Obtener qué celdas tienen otros compañeros para no duplicar en el selector
                celdas_ocupadas_otros = []
                for otro_op in ops_activos:
                    if otro_op != operario:
                        celdas_ocupadas_otros.extend(st.session_state.mapa_asignaciones.get(otro_op, []))
                
                opciones_disponibles_combo = sorted(list(set(maquinas_activas) - set(celdas_ocupadas_otros)))
                mis_celdas_actuales = [m for m in st.session_state.mapa_asignaciones.get(operario, []) if m in maquinas_activas]
                opciones_finales_combo = sorted(list(set(opciones_disponibles_combo) | set(mis_celdas_actuales)))

                # EL SELECTOR DE VERDAD: Lee y escribe directamente en la sesión global sin bloqueos
                nuevas_celdas = st.multiselect(
                    f"Asignar celdas:",
                    options=opciones_finales_combo,
                    default=mis_celdas_actuales,
                    key=f"combo_sync_v{st.session_state.sync_version}_{operario}"
                )
                
                # Sincronización inmediata de datos al persistir la acción del usuario
                st.session_state.mapa_asignaciones[operario] = nuevas_celdas

                # Cálculos matemáticos en vivo basados ESTRICTAMENTE en la selección actual
                carga_estatica = sum([cargas_dinamicas_turno.get(m, 0.0) for m in nuevas_celdas])
                carga_dinamica = calcular_carga_caminado(nuevas_celdas)
                carga_total_real = (carga_estatica + carga_dinamica) * 100.0

                st.markdown(f"**Carga Estática:** {carga_estatica*100:.1f}%")
                st.markdown(f"**Carga de Traslado:** {carga_dinamica*100:.1f}%")
                
                if carga_total_real > 110.0:
                    st.error(f"💥 Carga Total: {carga_total_real:.1f}%")
                elif carga_total_real > 95.0:
                    st.warning(f"⚠️ Carga Total: {carga_total_real:.1f}%")
                elif carga_total_real == 0.0:
                    st.info("Sin carga asignada")
                else:
                    st.success(f"⚡ Carga Total: {carga_total_real:.1f}%")

                if "904" in nuevas_celdas and "928" in nuevas_celdas:
                    st.error("🚨 RESTRICCIÓN: Combinación 928+904 inválida por distancia.")

# -------------------------------------------------------------------------
# BOTONERA DE CONTROL INFERIOR
# -------------------------------------------------------------------------
st.write("---")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🔄 Recalcular Optimización Dinámica (IA)", type="primary", use_container_width=True):
        aplicar_optimizacion_maestra()
        st.session_state.sync_version += 1
        st.rerun()

with col_btn2:
    components.html("""
        <button style="
            background-color: #24a0ed; border: none; color: white; padding: 11px 20px;
            text-align: center; text-decoration: none; display: inline-block; font-size: 16px;
            cursor: pointer; width: 100%; border-radius: 4px; font-family: sans-serif; font-weight: bold;
        " onclick="window.parent.print()">🖨️ Imprimir / Exportar Reporte de Turno</button>
    """, height=50)
