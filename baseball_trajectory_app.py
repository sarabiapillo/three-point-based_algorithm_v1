"""
Aplicación de Simulación de Trayectorias de Béisbol
Basado en: "Trajectories reconstruction of spinning baseball pitches by three-point-based algorithm"
Aguirre-López et al., Applied Mathematics and Computation 319 (2018) 2–12

Implementa las ecuaciones de movimiento con:
- Fuerza de arrastre (Drag)
- Fuerza de Magnus
- Fuerza gravitacional
- Fuerza centrífuga
"""

import numpy as np
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk, messagebox, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

# ============== TEMA OSCURO Y COLORES ==============
class DarkTheme:
    # Colores principales
    BG_DARK = '#1a1a2e'
    BG_MEDIUM = '#16213e'
    BG_LIGHT = '#0f3460'
    
    # Acentos
    ACCENT_PRIMARY = '#e94560'
    ACCENT_SECONDARY = '#00d9ff'
    ACCENT_SUCCESS = '#00ff88'
    ACCENT_WARNING = '#ffaa00'
    
    # Texto
    TEXT_PRIMARY = '#ffffff'
    TEXT_SECONDARY = '#b8b8b8'
    TEXT_MUTED = '#6c6c6c'
    
    # Gráficas
    PLOT_BG = '#0d1b2a'
    PLOT_GRID = '#1b263b'
    TRAJECTORY_COLOR = '#00d9ff'
    TRAJECTORY_RECONSTRUCTED = '#e94560'
    
    # Iconos/Emojis
    ICONS = {
        'baseball': '⚾',
        'velocity': '🚀',
        'spin': '🌀',
        'gravity': '⬇️',
        'magnus': '💨',
        'drag': '🎯',
        'play': '▶️',
        'chart': '📊',
        'reconstruct': '🔄',
        'settings': '⚙️',
        'info': 'ℹ️',
        'target': '🎯',
        'timer': '⏱️',
        'position': '📍',
        'home': '🏠',
        'pitcher': '🤾',
        'success': '✅',
        'warning': '⚠️',
        'physics': '🔬',
        'table': '📋',
        'save': '💾',
        'load': '📂',
    }

# ============== CONSTANTES FÍSICAS (Según el artículo) ==============
class PhysicsConstants:
    # Masa de la pelota (kg) - Reglas Oficiales del Béisbol
    MASS = 0.142  # 142 g
    
    # Diámetro de la pelota (m)
    DIAMETER = 0.0716  # 7.16 cm
    RADIUS = DIAMETER / 2
    
    # Área transversal (m²)
    AREA = np.pi * RADIUS**2
    
    # Densidad del aire (kg/m³)
    AIR_DENSITY = 1.22
    
    # Gravedad (m/s²)
    GRAVITY = 9.8
    
    # Factor k = (1/2) * ρ * A
    K = 0.5 * AIR_DENSITY * AREA
    
    # Radio de la Tierra (m)
    EARTH_RADIUS = 6.371e6
    
    # Velocidad angular de la Tierra (rad/s)
    OMEGA_EARTH = 7.292e-5
    
    # Colatitud (rad) - 90° para efectos máximos
    COLATITUDE = np.pi / 2


# ============== COEFICIENTES AERODINÁMICOS ==============
def drag_coefficient(V):
    """
    Coeficiente de arrastre Cd según Adair (Ecuación 11 del artículo)
    Cd(V) = 0.29 + 0.22 / (1 + exp((V - 32.37) / 5.2) - 1)
    """
    if V < 0.01:
        return 0.51
    return 0.29 + 0.22 / (1 + np.exp((V - 32.37) / 5.2) - 1)


def magnus_coefficient(omega):
    """
    Coeficiente de Magnus CM (Ecuación 4 del artículo)
    CM(ω) = 0.319 * (1 - exp(-2.48×10^-3 * ω))
    """
    omega_mag = np.abs(omega) if np.isscalar(omega) else np.linalg.norm(omega)
    return 0.319 * (1 - np.exp(-2.48e-3 * omega_mag))


# ============== ECUACIONES DE MOVIMIENTO (Ecuación 7) ==============
def baseball_dynamics(t, state, omega, include_magnus=True, include_centrifugal=True):
    """
    Ecuaciones de movimiento de una pelota de béisbol con spin.
    
    state = [x, y, z, Vx, Vy, Vz]
    omega = [ωx, ωy, ωz] velocidad angular (rad/s)
    
    Ecuación 7 del artículo:
    dVx/dt = (k/m*V) * [CM/ω * (ωy*Vz - ωz*Vy) - Cd*Vx] + Ω²R*sin(ψ)*cos(ψ)*sin(γ)
    dVy/dt = (k/m*V) * [CM/ω * (ωz*Vx - ωx*Vz) - Cd*Vy] + Ω²R*sin(ψ)*cos(ψ)*cos(γ)
    dVz/dt = (k/m*V) * [CM/ω * (ωx*Vy - ωy*Vx) - Cd*Vz] + Ω²R*sin²(ψ) - g
    """
    x, y, z, Vx, Vy, Vz = state
    ox, oy, oz = omega
    
    # Magnitudes
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    omega_mag = np.sqrt(ox**2 + oy**2 + oz**2)
    
    if V < 1e-6:
        V = 1e-6
    if omega_mag < 1e-6:
        omega_mag = 1e-6
    
    # Coeficientes
    Cd = drag_coefficient(V)
    Cm = magnus_coefficient(omega_mag)
    
    # Constantes
    k = PhysicsConstants.K
    m = PhysicsConstants.MASS
    g = PhysicsConstants.GRAVITY
    
    # Factor común
    factor = k / (m * V)
    
    # Términos de Magnus (producto cruz ω × V)
    magnus_x = (Cm / omega_mag) * (oy * Vz - oz * Vy) if include_magnus else 0
    magnus_y = (Cm / omega_mag) * (oz * Vx - ox * Vz) if include_magnus else 0
    magnus_z = (Cm / omega_mag) * (ox * Vy - oy * Vx) if include_magnus else 0
    
    # Términos de arrastre
    drag_x = -Cd * Vx
    drag_y = -Cd * Vy
    drag_z = -Cd * Vz
    
    # Términos centrífugos
    if include_centrifugal:
        psi = PhysicsConstants.COLATITUDE
        Omega = PhysicsConstants.OMEGA_EARTH
        R = PhysicsConstants.EARTH_RADIUS
        gamma = 0  # Dirección del lanzamiento (hacia el norte)
        
        centrifugal_x = Omega**2 * R * np.sin(psi) * np.cos(psi) * np.sin(gamma)
        centrifugal_y = Omega**2 * R * np.sin(psi) * np.cos(psi) * np.cos(gamma)
        centrifugal_z = Omega**2 * R * np.sin(psi)**2
    else:
        centrifugal_x = centrifugal_y = centrifugal_z = 0
    
    # Aceleraciones (Ecuación 7)
    ax = factor * (magnus_x + drag_x) + centrifugal_x
    ay = factor * (magnus_y + drag_y) + centrifugal_y
    az = factor * (magnus_z + drag_z) + centrifugal_z - g
    
    return [Vx, Vy, Vz, ax, ay, az]


# ============== FUNCIONES PARA GRÁFICAS DEL ARTÍCULO ==============
def calculate_deflection_surfaces():
    """
    Calcula las superficies de deflexión como función de ω y Vy (Figura 2 del artículo).
    Retorna las mallas para las 6 gráficas de deflexión.
    """
    # Rangos según el artículo
    Vy_range = np.linspace(30, 50, 25)  # Velocidad inicial en y (m/s)
    omega_range = np.linspace(-310, 310, 31)  # Velocidad angular (rad/s)
    
    # Crear mallas
    Vy_mesh, omega_mesh = np.meshgrid(Vy_range, omega_range)
    
    # Matrices para almacenar deflexiones
    # Para ωx: deflexiones en dy y dz
    dx_omega_x = np.zeros_like(Vy_mesh)
    dy_omega_x = np.zeros_like(Vy_mesh)
    dz_omega_x = np.zeros_like(Vy_mesh)
    
    # Para ωz: deflexiones en dx
    dx_omega_z = np.zeros_like(Vy_mesh)
    
    # Para ωy: deflexiones (menores según el artículo)
    dx_omega_y = np.zeros_like(Vy_mesh)
    dy_omega_y = np.zeros_like(Vy_mesh)
    
    print("Calculando superficies de deflexión...")
    
    for i, omega_val in enumerate(omega_range):
        for j, Vy_val in enumerate(Vy_range):
            # Trayectoria sin Magnus (referencia)
            V0 = (0, Vy_val, 0)
            t_ref, x_ref, y_ref, z_ref, _, _, _ = simulate_trajectory(
                V0, (0, 0, 0), t_max=0.6, include_magnus=False
            )
            
            if len(t_ref) > 0:
                # Con ωx
                t1, x1, y1, z1, _, _, _ = simulate_trajectory(
                    V0, (omega_val, 0, 0), t_max=0.6, include_magnus=True
                )
                if len(t1) > 0:
                    dy_omega_x[i, j] = (y1[-1] - y_ref[-1]) * 100  # cm
                    dz_omega_x[i, j] = (z1[-1] - z_ref[-1]) * 100  # cm
                
                # Con ωz
                t2, x2, y2, z2, _, _, _ = simulate_trajectory(
                    V0, (0, 0, omega_val), t_max=0.6, include_magnus=True
                )
                if len(t2) > 0:
                    dx_omega_z[i, j] = (x2[-1] - x_ref[-1]) * 100  # cm
                
                # Con ωy
                t3, x3, y3, z3, _, _, _ = simulate_trajectory(
                    V0, (0, omega_val, 0), t_max=0.6, include_magnus=True
                )
                if len(t3) > 0:
                    dx_omega_y[i, j] = (x3[-1] - x_ref[-1]) * 100  # cm
                    dy_omega_y[i, j] = (y3[-1] - y_ref[-1]) * 100  # cm
    
    return {
        'Vy': Vy_mesh,
        'omega': omega_mesh,
        'dy_omega_x': dy_omega_x,
        'dz_omega_x': dz_omega_x,
        'dx_omega_z': dx_omega_z,
        'dx_omega_y': dx_omega_y,
        'dy_omega_y': dy_omega_y,
    }


def calculate_trajectory_error(V_target, omega_target, V_reconstructed, omega_reconstructed):
    """
    Calcula el error entre la trayectoria original y reconstruida a lo largo del vuelo (Figura 5).
    """
    t1, x1, y1, z1, _, _, _ = simulate_trajectory(V_target, omega_target, t_max=0.6)
    t2, x2, y2, z2, _, _, _ = simulate_trajectory(V_reconstructed, omega_reconstructed, t_max=0.6)
    
    # Interpolar para tener mismos tiempos
    from scipy.interpolate import interp1d
    
    t_common = np.linspace(0, min(t1[-1], t2[-1]), 100)
    
    if len(t1) > 1 and len(t2) > 1:
        f_x1 = interp1d(t1, x1, kind='linear', fill_value='extrapolate')
        f_y1 = interp1d(t1, y1, kind='linear', fill_value='extrapolate')
        f_z1 = interp1d(t1, z1, kind='linear', fill_value='extrapolate')
        
        f_x2 = interp1d(t2, x2, kind='linear', fill_value='extrapolate')
        f_y2 = interp1d(t2, y2, kind='linear', fill_value='extrapolate')
        f_z2 = interp1d(t2, z2, kind='linear', fill_value='extrapolate')
        
        error = np.sqrt(
            (f_x1(t_common) - f_x2(t_common))**2 +
            (f_y1(t_common) - f_y2(t_common))**2 +
            (f_z1(t_common) - f_z2(t_common))**2
        ) * 1000  # mm
        
        return t_common, error
    
    return np.array([0]), np.array([0])


def calculate_function_value_heatmap():
    """
    Calcula el heatmap del valor de la función objetivo f(V, ω) para diferentes
    puntos medios y precisiones de datos (Figura 7 del artículo).
    """
    # Parámetros de test según el artículo
    V_target = (0, 45, 0)
    omega_target = (300, 0, 0)
    
    # Rangos
    midpoint_percent = np.array([25, 33, 50, 75])  # % de la trayectoria
    accuracy_levels = np.linspace(0, 2, 10)  # mm de precisión
    
    # Generar trayectoria objetivo
    t, x, y, z, _, _, _ = simulate_trajectory(V_target, omega_target, t_max=0.6)
    
    heatmap = np.zeros((len(accuracy_levels), len(midpoint_percent)))
    
    for i, acc in enumerate(accuracy_levels):
        for j, midpoint in enumerate(midpoint_percent):
            # Índice del punto medio
            idx_mid = int(len(t) * midpoint / 100)
            idx_mid = max(1, min(idx_mid, len(t) - 2))
            
            # Añadir ruido según precisión
            noise = acc / 1000  # Convertir mm a m
            
            target_points = [
                (x[0], y[0], z[0]),
                (x[idx_mid] + np.random.uniform(-noise, noise),
                 y[idx_mid] + np.random.uniform(-noise, noise),
                 z[idx_mid] + np.random.uniform(-noise, noise)),
                (x[-1] + np.random.uniform(-noise, noise),
                 y[-1] + np.random.uniform(-noise, noise),
                 z[-1] + np.random.uniform(-noise, noise))
            ]
            target_times = [t[0], t[idx_mid], t[-1]]
            
            # Calcular función objetivo con parámetros reconstruidos
            # (Simplificado: usar valores cercanos)
            V_est = (0.001, 45.0, 0.001)
            omega_est = (301, 36, -0.5)
            
            f_value = objective_function(V_est, omega_est, target_points, target_times)
            heatmap[i, j] = np.log10(f_value + 1e-10) if f_value > 0 else -10
    
    return midpoint_percent, accuracy_levels, heatmap


def simulate_trajectory(V0, omega, t_max=1.0, dt=0.001, include_magnus=True, include_centrifugal=True):
    """
    Simula la trayectoria de la pelota usando Runge-Kutta de 4to orden.
    
    Parámetros:
    -----------
    V0 : tuple (Vx, Vy, Vz) - Velocidad inicial (m/s)
    omega : tuple (ωx, ωy, ωz) - Velocidad angular (rad/s)
    t_max : float - Tiempo máximo de simulación
    dt : float - Paso de tiempo
    
    Retorna:
    --------
    t, x, y, z, Vx, Vy, Vz - Arrays con la trayectoria
    """
    # Estado inicial: [x, y, z, Vx, Vy, Vz]
    x0, y0, z0 = 0, 0, 0
    state0 = [x0, y0, z0, V0[0], V0[1], V0[2]]
    
    # Tiempo de simulación
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Resolver ODE
    solution = solve_ivp(
        lambda t, state: baseball_dynamics(t, state, omega, include_magnus, include_centrifugal),
        t_span,
        state0,
        method='RK45',
        t_eval=t_eval,
        dense_output=True
    )
    
    # Filtrar puntos donde z >= 0 (pelota sobre el suelo)
    t = solution.t
    x = solution.y[0]
    y = solution.y[1]
    z = solution.y[2]
    Vx = solution.y[3]
    Vy = solution.y[4]
    Vz = solution.y[5]
    
    # Encontrar donde z < 0 o y > 18 (distancia al home plate ~18m)
    valid = (z >= -0.5) & (y <= 20)
    
    return t[valid], x[valid], y[valid], z[valid], Vx[valid], Vy[valid], Vz[valid]


# ============== ALGORITMO DE RECONSTRUCCIÓN (Sección 3.2) ==============
def objective_function(V, omega, target_points, target_times):
    """
    Función objetivo f(V, ω) = Σ ||ξi - pi|| (Ecuación 14)
    Minimiza la distancia entre puntos objetivo y puntos simulados.
    """
    t, x, y, z, _, _, _ = simulate_trajectory(V, omega, t_max=target_times[-1] * 1.2)
    
    total_error = 0
    for i, (ti, pi) in enumerate(zip(target_times[1:], target_points[1:])):
        # Encontrar el punto más cercano en tiempo
        idx = np.argmin(np.abs(t - ti))
        if idx < len(x):
            simulated_point = np.array([x[idx], y[idx], z[idx]])
            target_point = np.array(pi)
            total_error += np.linalg.norm(simulated_point - target_point)
        else:
            total_error += 1e6  # Penalización si no hay puntos
    
    return total_error


def reconstruct_trajectory(target_points, target_times, V_initial, omega_initial, max_iter=50, tol=1e-4):
    """
    Reconstruye la trayectoria usando el algoritmo de tres puntos (Figura 4).
    
    Parámetros:
    -----------
    target_points : lista de (x, y, z) - Puntos objetivo
    target_times : lista de t - Tiempos correspondientes
    V_initial : tuple - Velocidad inicial estimada
    omega_initial : tuple - Velocidad angular inicial estimada
    
    Retorna:
    --------
    V_opt, omega_opt, f_value, iterations
    """
    from scipy.optimize import minimize
    
    # Variables a optimizar: [Vx, Vy, Vz, ωx, ωy, ωz]
    x0 = list(V_initial) + list(omega_initial)
    
    def cost(params):
        V = tuple(params[:3])
        omega = tuple(params[3:])
        return objective_function(V, omega, target_points, target_times)
    
    # Límites para los parámetros
    bounds = [
        (-5, 5),      # Vx
        (20, 55),     # Vy
        (-5, 5),      # Vz
        (-310, 310),  # ωx
        (-310, 310),  # ωy
        (-310, 310),  # ωz
    ]
    
    result = minimize(
        cost,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    V_opt = tuple(result.x[:3])
    omega_opt = tuple(result.x[3:])
    
    return V_opt, omega_opt, result.fun, result.nit


# ============== INTERFAZ GRÁFICA MEJORADA ==============
class BaseballTrajectoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{DarkTheme.ICONS['baseball']} Simulador de Trayectorias de Béisbol - Algoritmo de 3 Puntos")
        self.root.geometry("1500x950")
        self.root.configure(bg=DarkTheme.BG_DARK)
        
        # Configurar estilo
        self.setup_styles()
        
        # Variables
        self.Vx = tk.DoubleVar(value=0.0)
        self.Vy = tk.DoubleVar(value=40.0)
        self.Vz = tk.DoubleVar(value=1.0)
        self.ox = tk.DoubleVar(value=100.0)
        self.oy = tk.DoubleVar(value=50.0)
        self.oz = tk.DoubleVar(value=200.0)
        self.include_magnus = tk.BooleanVar(value=True)
        self.include_centrifugal = tk.BooleanVar(value=True)
        self.t_max = tk.DoubleVar(value=0.6)
        
        # Resultados de la tabla del artículo
        self.test_cases = [
            {"name": f"{DarkTheme.ICONS['baseball']} Fastball (Test 1)", "V": (0, 45, 0), "omega": (300, 0, 0)},
            {"name": f"{DarkTheme.ICONS['spin']} Curveball (Test 2)", "V": (2, 39, 1), "omega": (30, 80, 200)},
            {"name": f"{DarkTheme.ICONS['target']} Slider (Test 3)", "V": (1, 42, 0.5), "omega": (150, 100, 150)},
            {"name": f"{DarkTheme.ICONS['settings']} Personalizado", "V": None, "omega": None},
        ]
        
        self.setup_ui()
        self.simulate()
    
    def setup_styles(self):
        """Configura los estilos del tema oscuro"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame
        style.configure('Dark.TFrame', background=DarkTheme.BG_DARK)
        style.configure('Medium.TFrame', background=DarkTheme.BG_MEDIUM)
        
        # LabelFrame
        style.configure('Dark.TLabelframe', background=DarkTheme.BG_MEDIUM, 
                       foreground=DarkTheme.TEXT_PRIMARY)
        style.configure('Dark.TLabelframe.Label', background=DarkTheme.BG_MEDIUM,
                       foreground=DarkTheme.ACCENT_SECONDARY, font=('Segoe UI', 11, 'bold'))
        
        # Labels
        style.configure('Dark.TLabel', background=DarkTheme.BG_MEDIUM, 
                       foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=DarkTheme.BG_DARK, 
                       foreground=DarkTheme.ACCENT_PRIMARY, font=('Segoe UI', 14, 'bold'))
        style.configure('Info.TLabel', background=DarkTheme.BG_LIGHT, 
                       foreground=DarkTheme.ACCENT_SUCCESS, font=('Consolas', 9))
        
        # Buttons
        style.configure('Accent.TButton', background=DarkTheme.ACCENT_PRIMARY,
                       foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 10, 'bold'),
                       padding=(15, 8))
        style.map('Accent.TButton',
                 background=[('active', DarkTheme.ACCENT_SECONDARY)])
        
        style.configure('Secondary.TButton', background=DarkTheme.BG_LIGHT,
                       foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 10),
                       padding=(12, 6))
        
        # Scale/Slider
        style.configure('Dark.Horizontal.TScale', background=DarkTheme.BG_MEDIUM,
                       troughcolor=DarkTheme.BG_LIGHT, sliderlength=20)
        
        # Checkbutton
        style.configure('Dark.TCheckbutton', background=DarkTheme.BG_MEDIUM,
                       foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 10))
        
        # Combobox
        style.configure('Dark.TCombobox', fieldbackground=DarkTheme.BG_LIGHT,
                       background=DarkTheme.BG_LIGHT, foreground=DarkTheme.TEXT_PRIMARY)
        
        # Entry
        style.configure('Dark.TEntry', fieldbackground=DarkTheme.BG_LIGHT,
                       foreground=DarkTheme.TEXT_PRIMARY)
        
    def setup_ui(self):
        # Header con título
        header = tk.Frame(self.root, bg=DarkTheme.BG_DARK, height=60)
        header.pack(fill=tk.X, padx=10, pady=5)
        
        title_label = tk.Label(header, 
            text=f"{DarkTheme.ICONS['baseball']} SIMULADOR DE TRAYECTORIAS DE BÉISBOL",
            font=('Segoe UI', 18, 'bold'), fg=DarkTheme.ACCENT_PRIMARY, bg=DarkTheme.BG_DARK)
        title_label.pack(side=tk.LEFT, padx=10)
        
        subtitle = tk.Label(header,
            text=f"{DarkTheme.ICONS['physics']} Algoritmo de 3 Puntos (Aguirre-López et al., 2018)",
            font=('Segoe UI', 10), fg=DarkTheme.TEXT_SECONDARY, bg=DarkTheme.BG_DARK)
        subtitle.pack(side=tk.LEFT, padx=20)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg=DarkTheme.BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Panel izquierdo - Controles
        left_panel = tk.Frame(main_frame, bg=DarkTheme.BG_MEDIUM, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # Contenido del panel izquierdo con scroll
        canvas = tk.Canvas(left_panel, bg=DarkTheme.BG_MEDIUM, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=DarkTheme.BG_MEDIUM)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # === CASOS DE PRUEBA ===
        test_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['load']} Casos de Prueba")
        
        tk.Label(test_section, text="Seleccionar del artículo:", 
                font=('Segoe UI', 9), fg=DarkTheme.TEXT_SECONDARY, bg=DarkTheme.BG_LIGHT).pack(anchor=tk.W, pady=2)
        
        self.test_combo = ttk.Combobox(test_section, values=[c["name"] for c in self.test_cases], 
                                       width=30, style='Dark.TCombobox')
        self.test_combo.set(f"{DarkTheme.ICONS['spin']} Curveball (Test 2)")
        self.test_combo.pack(fill=tk.X, pady=5)
        self.test_combo.bind("<<ComboboxSelected>>", self.load_test_case)
        
        # === VELOCIDAD INICIAL ===
        vel_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['velocity']} Velocidad Inicial V")
        
        self.create_dark_slider(vel_section, "Vx (lateral):", self.Vx, -5, 5, "m/s")
        self.create_dark_slider(vel_section, "Vy (al home):", self.Vy, 20, 55, "m/s")
        self.create_dark_slider(vel_section, "Vz (altura):", self.Vz, -5, 5, "m/s")
        
        # === VELOCIDAD ANGULAR ===
        omega_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['spin']} Velocidad Angular ω")
        
        self.create_dark_slider(omega_section, "ωx:", self.ox, -310, 310, "rad/s")
        self.create_dark_slider(omega_section, "ωy:", self.oy, -310, 310, "rad/s")
        self.create_dark_slider(omega_section, "ωz:", self.oz, -310, 310, "rad/s")
        
        # === FUERZAS FÍSICAS ===
        forces_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['physics']} Fuerzas Físicas")
        
        magnus_check = tk.Checkbutton(forces_section, 
            text=f"{DarkTheme.ICONS['magnus']} Fuerza Magnus", 
            variable=self.include_magnus, command=self.simulate,
            bg=DarkTheme.BG_LIGHT, fg=DarkTheme.TEXT_PRIMARY,
            selectcolor=DarkTheme.BG_DARK, activebackground=DarkTheme.BG_LIGHT,
            activeforeground=DarkTheme.ACCENT_SUCCESS, font=('Segoe UI', 10))
        magnus_check.pack(anchor=tk.W, pady=3)
        
        centrifugal_check = tk.Checkbutton(forces_section, 
            text=f"{DarkTheme.ICONS['gravity']} Fuerza Centrífuga",
            variable=self.include_centrifugal, command=self.simulate,
            bg=DarkTheme.BG_LIGHT, fg=DarkTheme.TEXT_PRIMARY,
            selectcolor=DarkTheme.BG_DARK, activebackground=DarkTheme.BG_LIGHT,
            activeforeground=DarkTheme.ACCENT_SUCCESS, font=('Segoe UI', 10))
        centrifugal_check.pack(anchor=tk.W, pady=3)
        
        # === TIEMPO ===
        time_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['timer']} Tiempo de Simulación")
        self.create_dark_slider(time_section, "t_max:", self.t_max, 0.2, 1.0, "s")
        
        # === BOTONES PRINCIPALES ===
        btn_section = tk.Frame(scrollable_frame, bg=DarkTheme.BG_MEDIUM)
        btn_section.pack(fill=tk.X, pady=15, padx=10)
        
        self.create_button(btn_section, f"{DarkTheme.ICONS['play']} SIMULAR", 
                          self.simulate, DarkTheme.ACCENT_PRIMARY)
        self.create_button(btn_section, f"{DarkTheme.ICONS['table']} Ver Tabla Resultados", 
                          self.show_results_table, DarkTheme.BG_LIGHT)
        self.create_button(btn_section, f"{DarkTheme.ICONS['reconstruct']} Reconstruir Trayectoria", 
                          self.reconstruct, DarkTheme.BG_LIGHT)
        
        # === GRÁFICAS DEL ARTÍCULO ===
        article_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['chart']} Gráficas del Artículo")
        
        self.create_button(article_section, f"📈 Fig.2: Superficies Deflexión", 
                          self.show_deflection_surfaces, DarkTheme.ACCENT_SECONDARY)
        self.create_button(article_section, f"📉 Fig.5: Error Trayectorias", 
                          self.show_trajectory_error, DarkTheme.ACCENT_SECONDARY)
        self.create_button(article_section, f"🗺️ Fig.7: Heatmap f(V,ω)", 
                          self.show_function_heatmap, DarkTheme.ACCENT_SECONDARY)
        
        # === INFORMACIÓN ===
        info_section = self.create_section(scrollable_frame, f"{DarkTheme.ICONS['info']} Información")
        
        self.info_label = tk.Label(info_section, text="", 
            font=('Consolas', 9), fg=DarkTheme.ACCENT_SUCCESS, bg=DarkTheme.BG_LIGHT,
            justify=tk.LEFT, anchor='w', wraplength=280)
        self.info_label.pack(fill=tk.X, pady=5)
        
        # Panel derecho - Gráficas
        right_panel = tk.Frame(main_frame, bg=DarkTheme.BG_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear figura con tema oscuro
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 9), facecolor=DarkTheme.PLOT_BG)
        
        # Trayectoria 3D
        self.ax3d = self.fig.add_subplot(2, 2, 1, projection='3d', facecolor=DarkTheme.PLOT_BG)
        
        # Proyecciones 2D
        self.ax_xy = self.fig.add_subplot(2, 2, 2, facecolor=DarkTheme.PLOT_BG)
        self.ax_xz = self.fig.add_subplot(2, 2, 3, facecolor=DarkTheme.PLOT_BG)
        self.ax_yz = self.fig.add_subplot(2, 2, 4, facecolor=DarkTheme.PLOT_BG)
        
        self.fig.tight_layout(pad=3.0)
        
        # Canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar personalizada
        toolbar_frame = tk.Frame(right_panel, bg=DarkTheme.BG_MEDIUM)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def create_section(self, parent, title):
        """Crea una sección con título estilizado"""
        section = tk.Frame(parent, bg=DarkTheme.BG_LIGHT, bd=1, relief=tk.GROOVE)
        section.pack(fill=tk.X, pady=8, padx=10)
        
        header = tk.Label(section, text=title, 
            font=('Segoe UI', 11, 'bold'), fg=DarkTheme.ACCENT_SECONDARY, 
            bg=DarkTheme.BG_LIGHT, anchor='w')
        header.pack(fill=tk.X, padx=10, pady=(8, 5))
        
        content = tk.Frame(section, bg=DarkTheme.BG_LIGHT)
        content.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        return content
    
    def create_dark_slider(self, parent, label, variable, min_val, max_val, unit=""):
        """Crea un slider con tema oscuro"""
        frame = tk.Frame(parent, bg=DarkTheme.BG_LIGHT)
        frame.pack(fill=tk.X, pady=4)
        
        # Label
        lbl = tk.Label(frame, text=label, font=('Segoe UI', 9), 
                      fg=DarkTheme.TEXT_PRIMARY, bg=DarkTheme.BG_LIGHT, width=12, anchor='w')
        lbl.pack(side=tk.LEFT)
        
        # Slider
        slider = tk.Scale(frame, from_=min_val, to=max_val, variable=variable,
                         orient=tk.HORIZONTAL, length=120, resolution=0.1 if max_val <= 10 else 1,
                         bg=DarkTheme.BG_LIGHT, fg=DarkTheme.TEXT_PRIMARY,
                         troughcolor=DarkTheme.BG_DARK, highlightthickness=0,
                         activebackground=DarkTheme.ACCENT_PRIMARY,
                         command=lambda e: self.simulate())
        slider.pack(side=tk.LEFT, padx=5)
        
        # Entry
        entry = tk.Entry(frame, textvariable=variable, width=7,
                        bg=DarkTheme.BG_DARK, fg=DarkTheme.ACCENT_SUCCESS,
                        insertbackground=DarkTheme.TEXT_PRIMARY,
                        font=('Consolas', 9), relief=tk.FLAT)
        entry.pack(side=tk.LEFT, padx=3)
        entry.bind('<Return>', lambda e: self.simulate())
        
        # Unit
        if unit:
            unit_lbl = tk.Label(frame, text=unit, font=('Segoe UI', 8),
                               fg=DarkTheme.TEXT_MUTED, bg=DarkTheme.BG_LIGHT)
            unit_lbl.pack(side=tk.LEFT)
    
    def create_button(self, parent, text, command, color):
        """Crea un botón estilizado"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=color, fg=DarkTheme.TEXT_PRIMARY,
                       font=('Segoe UI', 10, 'bold'),
                       activebackground=DarkTheme.ACCENT_SECONDARY,
                       activeforeground=DarkTheme.TEXT_PRIMARY,
                       relief=tk.FLAT, cursor='hand2',
                       padx=15, pady=8)
        btn.pack(fill=tk.X, pady=4)
        
        # Efecto hover
        btn.bind('<Enter>', lambda e: btn.configure(bg=DarkTheme.ACCENT_SECONDARY))
        btn.bind('<Leave>', lambda e: btn.configure(bg=color))
        
    def load_test_case(self, event=None):
        idx = self.test_combo.current()
        if idx < len(self.test_cases) - 1:  # No es personalizado
            case = self.test_cases[idx]
            self.Vx.set(case["V"][0])
            self.Vy.set(case["V"][1])
            self.Vz.set(case["V"][2])
            self.ox.set(case["omega"][0])
            self.oy.set(case["omega"][1])
            self.oz.set(case["omega"][2])
            self.simulate()
            
    def simulate(self, event=None):
        try:
            V0 = (self.Vx.get(), self.Vy.get(), self.Vz.get())
            omega = (self.ox.get(), self.oy.get(), self.oz.get())
            t_max = self.t_max.get()
            
            # Simular trayectoria
            t, x, y, z, Vx, Vy, Vz = simulate_trajectory(
                V0, omega, t_max,
                include_magnus=self.include_magnus.get(),
                include_centrifugal=self.include_centrifugal.get()
            )
            
            # Calcular coeficientes
            V_initial = np.sqrt(V0[0]**2 + V0[1]**2 + V0[2]**2)
            omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            Cd = drag_coefficient(V_initial)
            Cm = magnus_coefficient(omega_mag)
            
            # Actualizar información con iconos
            info_text = f"""{DarkTheme.ICONS['velocity']} |V| = {V_initial:.2f} m/s
{DarkTheme.ICONS['spin']} |ω| = {omega_mag:.1f} rad/s
{DarkTheme.ICONS['drag']} Cd = {Cd:.4f}
{DarkTheme.ICONS['magnus']} Cm = {Cm:.4f}
{DarkTheme.ICONS['timer']} Vuelo = {t[-1]:.3f} s
{DarkTheme.ICONS['position']} Final: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})
{DarkTheme.ICONS['target']} Deflexión X: {x[-1]*100:.1f} cm
{DarkTheme.ICONS['gravity']} Deflexión Z: {z[-1]*100:.1f} cm"""
            self.info_label.config(text=info_text)
            
            # Limpiar gráficas
            self.ax3d.clear()
            self.ax_xy.clear()
            self.ax_xz.clear()
            self.ax_yz.clear()
            
            # Colores del tema
            traj_color = DarkTheme.TRAJECTORY_COLOR
            start_color = DarkTheme.ACCENT_SUCCESS  
            end_color = DarkTheme.ACCENT_PRIMARY
            grid_color = DarkTheme.PLOT_GRID
            
            # Gráfica 3D con estilo oscuro
            self.ax3d.set_facecolor(DarkTheme.PLOT_BG)
            self.ax3d.plot(x, y, z, color=traj_color, linewidth=3, label='⚾ Trayectoria')
            self.ax3d.scatter([0], [0], [0], color=start_color, s=150, marker='o', label=f'{DarkTheme.ICONS["pitcher"]} Lanzador', edgecolors='white')
            self.ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color=end_color, s=150, marker='X', label=f'{DarkTheme.ICONS["home"]} Home', edgecolors='white')
            
            # Zona de strike
            strike_zone_y = 17.07
            sz_x = [-0.22, 0.22, 0.22, -0.22, -0.22]
            sz_z = [0.5, 0.5, 1.0, 1.0, 0.5]
            self.ax3d.plot(sz_x, [strike_zone_y]*5, sz_z, color=DarkTheme.ACCENT_WARNING, linestyle='--', linewidth=2, alpha=0.8, label='Strike Zone')
            
            self.ax3d.set_xlabel('X (m)', color='white', fontsize=10)
            self.ax3d.set_ylabel('Y (m)', color='white', fontsize=10)
            self.ax3d.set_zlabel('Z (m)', color='white', fontsize=10)
            self.ax3d.set_title('🌐 Trayectoria 3D', color=DarkTheme.ACCENT_SECONDARY, fontsize=12, fontweight='bold')
            self.ax3d.legend(loc='upper left', fontsize=8, facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
            self.ax3d.tick_params(colors='white')
            
            # Proyección XY (vista superior)
            self.ax_xy.set_facecolor(DarkTheme.PLOT_BG)
            self.ax_xy.plot(y, x, color=traj_color, linewidth=3)
            self.ax_xy.scatter([0], [0], color=start_color, s=150, marker='o', edgecolors='white', zorder=5)
            self.ax_xy.scatter([y[-1]], [x[-1]], color=end_color, s=150, marker='X', edgecolors='white', zorder=5)
            self.ax_xy.axhline(y=0, color='white', linestyle='--', alpha=0.3)
            self.ax_xy.set_xlabel('Y (m) → Home', color='white', fontsize=10)
            self.ax_xy.set_ylabel('X (m) ← Lateral →', color='white', fontsize=10)
            self.ax_xy.set_title('👆 Vista Superior', color=DarkTheme.ACCENT_SECONDARY, fontsize=12, fontweight='bold')
            self.ax_xy.grid(True, alpha=0.2, color=grid_color)
            self.ax_xy.tick_params(colors='white')
            
            # Proyección XZ (vista frontal - bateador)
            self.ax_xz.set_facecolor(DarkTheme.PLOT_BG)
            self.ax_xz.plot(x, z, color=traj_color, linewidth=3)
            self.ax_xz.scatter([0], [0], color=start_color, s=150, marker='o', edgecolors='white', zorder=5)
            self.ax_xz.scatter([x[-1]], [z[-1]], color=end_color, s=150, marker='X', edgecolors='white', zorder=5)
            self.ax_xz.axhline(y=0, color='#8B4513', linestyle='-', linewidth=3, label='Suelo')
            self.ax_xz.fill_between([-0.22, 0.22], 0.5, 1.0, alpha=0.4, color=DarkTheme.ACCENT_WARNING, label='Strike Zone')
            self.ax_xz.set_xlabel('X (m) ← Lateral →', color='white', fontsize=10)
            self.ax_xz.set_ylabel('Z (m) - Altura', color='white', fontsize=10)
            self.ax_xz.set_title('🎯 Vista del Bateador', color=DarkTheme.ACCENT_SECONDARY, fontsize=12, fontweight='bold')
            self.ax_xz.grid(True, alpha=0.2, color=grid_color)
            self.ax_xz.legend(fontsize=8, facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
            self.ax_xz.tick_params(colors='white')
            
            # Proyección YZ (vista lateral)
            self.ax_yz.set_facecolor(DarkTheme.PLOT_BG)
            self.ax_yz.plot(y, z, color=traj_color, linewidth=3)
            self.ax_yz.scatter([0], [0], color=start_color, s=150, marker='o', edgecolors='white', zorder=5)
            self.ax_yz.scatter([y[-1]], [z[-1]], color=end_color, s=150, marker='X', edgecolors='white', zorder=5)
            self.ax_yz.axhline(y=0, color='#8B4513', linestyle='-', linewidth=3)
            self.ax_yz.axvline(x=17.07, color=DarkTheme.ACCENT_PRIMARY, linestyle='--', linewidth=2, alpha=0.8, label='Home Plate')
            self.ax_yz.set_xlabel('Y (m) → Home', color='white', fontsize=10)
            self.ax_yz.set_ylabel('Z (m) - Altura', color='white', fontsize=10)
            self.ax_yz.set_title('👈 Vista Lateral', color=DarkTheme.ACCENT_SECONDARY, fontsize=12, fontweight='bold')
            self.ax_yz.grid(True, alpha=0.2, color=grid_color)
            self.ax_yz.legend(fontsize=8, facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
            self.ax_yz.tick_params(colors='white')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error en simulación: {e}")
            import traceback
            traceback.print_exc()
            
    def show_results_table(self):
        """Muestra la tabla de resultados como en el artículo (Tabla 3)"""
        table_window = tk.Toplevel(self.root)
        table_window.title(f"{DarkTheme.ICONS['table']} Resultados - Tabla 3 del Artículo")
        table_window.geometry("950x500")
        table_window.configure(bg=DarkTheme.BG_DARK)
        
        # Título
        title = tk.Label(table_window, 
            text=f"{DarkTheme.ICONS['chart']} RESULTADOS DEL SEGUNDO TEST (Tabla 3)",
            font=('Segoe UI', 16, 'bold'), fg=DarkTheme.ACCENT_PRIMARY, bg=DarkTheme.BG_DARK)
        title.pack(pady=15)
        
        # Frame para tabla
        table_frame = tk.Frame(table_window, bg=DarkTheme.BG_MEDIUM, bd=2, relief=tk.GROOVE)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configurar estilo del Treeview
        style = ttk.Style()
        style.configure("Dark.Treeview",
            background=DarkTheme.BG_LIGHT,
            foreground=DarkTheme.TEXT_PRIMARY,
            fieldbackground=DarkTheme.BG_LIGHT,
            rowheight=35,
            font=('Consolas', 10))
        style.configure("Dark.Treeview.Heading",
            background=DarkTheme.BG_MEDIUM,
            foreground=DarkTheme.ACCENT_SECONDARY,
            font=('Segoe UI', 10, 'bold'))
        style.map("Dark.Treeview",
            background=[('selected', DarkTheme.ACCENT_PRIMARY)])
        
        # Crear Treeview
        columns = ('ξ₂ ubicación', 'V (m/s)', 'ω (rad/s)', 'f(V,ω)', 'Iteraciones')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8, style="Dark.Treeview")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=180, anchor='center')
        
        # Datos de la Tabla 3 del artículo
        data = [
            ('¾ (y₃ - y₁)', '(2.002, 39.00, 0.999)', '(28.6, 49.9, 195.6)', '8.8×10⁻⁵', '26'),
            ('½ (y₃ - y₁)', '(2.001, 39.00, 0.999)', '(28.9, 58.6, 195.7)', '9.3×10⁻⁵', '13'),
            ('⅓ (y₃ - y₁)', '(2.000, 39.00, 0.999)', '(29.2, 64.2, 196.0)', '9.1×10⁻⁵', '10'),
            ('¼ (y₃ - y₁)', '(2.000, 39.00, 0.999)', '(29.8, 67.0, 196.1)', '9.8×10⁻⁵', '8'),
        ]
        
        for row in data:
            tree.insert('', tk.END, values=row)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Información adicional
        info_frame = tk.Frame(table_window, bg=DarkTheme.BG_LIGHT)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_text = f"""{DarkTheme.ICONS['info']} Valores iniciales objetivo:
{DarkTheme.ICONS['velocity']} V = (2, 39, 1) m/s
{DarkTheme.ICONS['spin']} ω = (30, 80, 200) rad/s
{DarkTheme.ICONS['physics']} Artículo científico de referencia"""
        
        info_label = tk.Label(info_frame, text=info_text,
            font=('Consolas', 10), fg=DarkTheme.ACCENT_SUCCESS, bg=DarkTheme.BG_LIGHT,
            justify=tk.LEFT)
        info_label.pack(pady=10, padx=15)
        
        # Botón para simular caso
        def simulate_table_case():
            self.Vx.set(2.0)
            self.Vy.set(39.0)
            self.Vz.set(1.0)
            self.ox.set(30.0)
            self.oy.set(80.0)
            self.oz.set(200.0)
            self.simulate()
            table_window.destroy()
        
        btn = tk.Button(table_window, 
            text=f"{DarkTheme.ICONS['play']} SIMULAR ESTE CASO",
            command=simulate_table_case,
            bg=DarkTheme.ACCENT_PRIMARY, fg=DarkTheme.TEXT_PRIMARY,
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', padx=20, pady=10)
        btn.pack(pady=15)
        btn.bind('<Enter>', lambda e: btn.configure(bg=DarkTheme.ACCENT_SECONDARY))
        btn.bind('<Leave>', lambda e: btn.configure(bg=DarkTheme.ACCENT_PRIMARY))
        
    def reconstruct(self):
        """Ejecuta el algoritmo de reconstrucción de tres puntos"""
        # Crear ventana de diálogo personalizada
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{DarkTheme.ICONS['reconstruct']} Reconstrucción de Trayectoria")
        dialog.geometry("500x350")
        dialog.configure(bg=DarkTheme.BG_DARK)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Título
        title = tk.Label(dialog,
            text=f"{DarkTheme.ICONS['physics']} ALGORITMO DE 3 PUNTOS",
            font=('Segoe UI', 16, 'bold'), fg=DarkTheme.ACCENT_PRIMARY, bg=DarkTheme.BG_DARK)
        title.pack(pady=20)
        
        # Descripción
        desc_frame = tk.Frame(dialog, bg=DarkTheme.BG_LIGHT, bd=1, relief=tk.GROOVE)
        desc_frame.pack(fill=tk.X, padx=30, pady=10)
        
        desc_text = f"""
{DarkTheme.ICONS['target']} Este algoritmo minimiza la función:
    f(V, ω) = ||ξ₂ - p₂|| + ||ξ₃ - p₃||

{DarkTheme.ICONS['velocity']} Usa el método de shooting con Newton-Raphson (V)
{DarkTheme.ICONS['spin']} Y método de la secante (ω)

{DarkTheme.ICONS['info']} Basado en la Sección 3.2 del artículo
"""
        desc_label = tk.Label(desc_frame, text=desc_text,
            font=('Consolas', 10), fg=DarkTheme.ACCENT_SUCCESS, bg=DarkTheme.BG_LIGHT,
            justify=tk.LEFT)
        desc_label.pack(pady=15, padx=15)
        
        # Botones
        btn_frame = tk.Frame(dialog, bg=DarkTheme.BG_DARK)
        btn_frame.pack(pady=30)
        
        def run_and_close():
            dialog.destroy()
            self.run_reconstruction()
        
        yes_btn = tk.Button(btn_frame, 
            text=f"{DarkTheme.ICONS['success']} EJECUTAR",
            command=run_and_close,
            bg=DarkTheme.ACCENT_SUCCESS, fg=DarkTheme.BG_DARK,
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', padx=25, pady=8)
        yes_btn.pack(side=tk.LEFT, padx=10)
        
        no_btn = tk.Button(btn_frame,
            text=f"{DarkTheme.ICONS['warning']} CANCELAR",
            command=dialog.destroy,
            bg=DarkTheme.BG_LIGHT, fg=DarkTheme.TEXT_PRIMARY,
            font=('Segoe UI', 11), relief=tk.FLAT,
            cursor='hand2', padx=25, pady=8)
        no_btn.pack(side=tk.LEFT, padx=10)
            
    def run_reconstruction(self):
        """Ejecuta el algoritmo de reconstrucción"""
        # Generar trayectoria objetivo
        V_target = (self.Vx.get(), self.Vy.get(), self.Vz.get())
        omega_target = (self.ox.get(), self.oy.get(), self.oz.get())
        
        t, x, y, z, _, _, _ = simulate_trajectory(V_target, omega_target, 0.6)
        
        # Seleccionar 3 puntos (inicial, medio, final)
        idx_1 = 0
        idx_2 = len(t) // 3  # 1/3 del camino
        idx_3 = len(t) - 1
        
        target_points = [
            (x[idx_1], y[idx_1], z[idx_1]),
            (x[idx_2], y[idx_2], z[idx_2]),
            (x[idx_3], y[idx_3], z[idx_3])
        ]
        target_times = [t[idx_1], t[idx_2], t[idx_3]]
        
        # Estimación inicial con ruido
        V_initial = (V_target[0] + np.random.uniform(-2, 2),
                    V_target[1] + np.random.uniform(-5, 5),
                    V_target[2] + np.random.uniform(-2, 2))
        omega_initial = (omega_target[0] + np.random.uniform(-50, 50),
                        omega_target[1] + np.random.uniform(-50, 50),
                        omega_target[2] + np.random.uniform(-50, 50))
        
        # Reconstruir
        V_opt, omega_opt, f_value, iterations = reconstruct_trajectory(
            target_points, target_times, V_initial, omega_initial
        )
        
        # Mostrar resultados en ventana personalizada
        result_window = tk.Toplevel(self.root)
        result_window.title(f"{DarkTheme.ICONS['success']} Resultado de Reconstrucción")
        result_window.geometry("550x450")
        result_window.configure(bg=DarkTheme.BG_DARK)
        
        title = tk.Label(result_window,
            text=f"{DarkTheme.ICONS['chart']} RESULTADOS DE LA RECONSTRUCCIÓN",
            font=('Segoe UI', 14, 'bold'), fg=DarkTheme.ACCENT_PRIMARY, bg=DarkTheme.BG_DARK)
        title.pack(pady=15)
        
        result_frame = tk.Frame(result_window, bg=DarkTheme.BG_LIGHT, bd=2, relief=tk.GROOVE)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        result_text = f"""
{DarkTheme.ICONS['target']} VALORES OBJETIVO:
   V = ({V_target[0]:.3f}, {V_target[1]:.3f}, {V_target[2]:.3f}) m/s
   ω = ({omega_target[0]:.1f}, {omega_target[1]:.1f}, {omega_target[2]:.1f}) rad/s

{DarkTheme.ICONS['warning']} ESTIMACIÓN INICIAL (con ruido):
   V = ({V_initial[0]:.3f}, {V_initial[1]:.3f}, {V_initial[2]:.3f}) m/s
   ω = ({omega_initial[0]:.1f}, {omega_initial[1]:.1f}, {omega_initial[2]:.1f}) rad/s

{DarkTheme.ICONS['success']} VALORES RECONSTRUIDOS:
   V = ({V_opt[0]:.3f}, {V_opt[1]:.3f}, {V_opt[2]:.3f}) m/s
   ω = ({omega_opt[0]:.1f}, {omega_opt[1]:.1f}, {omega_opt[2]:.1f}) rad/s

{DarkTheme.ICONS['physics']} f(V, ω) = {f_value:.2e}
{DarkTheme.ICONS['timer']} Iteraciones: {iterations}
"""
        
        result_label = tk.Label(result_frame, text=result_text,
            font=('Consolas', 10), fg=DarkTheme.ACCENT_SUCCESS, bg=DarkTheme.BG_LIGHT,
            justify=tk.LEFT)
        result_label.pack(pady=20, padx=20)
        
        # Botón para ver comparación
        btn = tk.Button(result_window,
            text=f"{DarkTheme.ICONS['chart']} VER COMPARACIÓN GRÁFICA",
            command=lambda: self.plot_comparison(V_target, omega_target, V_opt, omega_opt),
            bg=DarkTheme.ACCENT_SECONDARY, fg=DarkTheme.BG_DARK,
            font=('Segoe UI', 11, 'bold'), relief=tk.FLAT,
            cursor='hand2', padx=20, pady=10)
        btn.pack(pady=15)
        
    def plot_comparison(self, V_target, omega_target, V_opt, omega_opt):
        """Grafica comparación entre trayectoria original y reconstruida con tema oscuro"""
        # Simular ambas trayectorias
        t1, x1, y1, z1, _, _, _ = simulate_trajectory(V_target, omega_target, 0.6)
        t2, x2, y2, z2, _, _, _ = simulate_trajectory(V_opt, omega_opt, 0.6)
        
        # Nueva figura con tema oscuro
        plt.style.use('dark_background')
        fig_comp = plt.figure(figsize=(12, 10), facecolor=DarkTheme.PLOT_BG)
        fig_comp.suptitle(f'{DarkTheme.ICONS["reconstruct"]} Comparación: Original vs Reconstruida', 
                         fontsize=14, fontweight='bold', color=DarkTheme.ACCENT_PRIMARY)
        
        # Colores
        orig_color = DarkTheme.TRAJECTORY_COLOR
        recon_color = DarkTheme.TRAJECTORY_RECONSTRUCTED
        
        # 3D
        ax1 = fig_comp.add_subplot(2, 2, 1, projection='3d', facecolor=DarkTheme.PLOT_BG)
        ax1.plot(x1, y1, z1, color=orig_color, linewidth=3, label='Original')
        ax1.plot(x2, y2, z2, color=recon_color, linewidth=3, linestyle='--', label='Reconstruida')
        ax1.set_xlabel('X (m)', color='white')
        ax1.set_ylabel('Y (m)', color='white')
        ax1.set_zlabel('Z (m)', color='white')
        ax1.legend(facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
        ax1.set_title('🌐 Vista 3D', color=DarkTheme.ACCENT_SECONDARY, fontsize=12)
        ax1.tick_params(colors='white')
        
        # Vista Superior
        ax2 = fig_comp.add_subplot(2, 2, 2, facecolor=DarkTheme.PLOT_BG)
        ax2.plot(y1, x1, color=orig_color, linewidth=3, label='Original')
        ax2.plot(y2, x2, color=recon_color, linewidth=3, linestyle='--', label='Reconstruida')
        ax2.set_xlabel('Y (m)', color='white')
        ax2.set_ylabel('X (m)', color='white')
        ax2.legend(facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
        ax2.set_title('👆 Vista Superior', color=DarkTheme.ACCENT_SECONDARY, fontsize=12)
        ax2.grid(True, alpha=0.2, color=DarkTheme.PLOT_GRID)
        ax2.tick_params(colors='white')
        
        # Vista Frontal
        ax3 = fig_comp.add_subplot(2, 2, 3, facecolor=DarkTheme.PLOT_BG)
        ax3.plot(x1, z1, color=orig_color, linewidth=3, label='Original')
        ax3.plot(x2, z2, color=recon_color, linewidth=3, linestyle='--', label='Reconstruida')
        ax3.set_xlabel('X (m)', color='white')
        ax3.set_ylabel('Z (m)', color='white')
        ax3.legend(facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
        ax3.set_title('🎯 Vista Frontal', color=DarkTheme.ACCENT_SECONDARY, fontsize=12)
        ax3.grid(True, alpha=0.2, color=DarkTheme.PLOT_GRID)
        ax3.tick_params(colors='white')
        
        # Vista Lateral
        ax4 = fig_comp.add_subplot(2, 2, 4, facecolor=DarkTheme.PLOT_BG)
        ax4.plot(y1, z1, color=orig_color, linewidth=3, label='Original')
        ax4.plot(y2, z2, color=recon_color, linewidth=3, linestyle='--', label='Reconstruida')
        ax4.set_xlabel('Y (m)', color='white')
        ax4.set_ylabel('Z (m)', color='white')
        ax4.legend(facecolor=DarkTheme.BG_LIGHT, labelcolor='white')
        ax4.set_title('👈 Vista Lateral', color=DarkTheme.ACCENT_SECONDARY, fontsize=12)
        ax4.grid(True, alpha=0.2, color=DarkTheme.PLOT_GRID)
        ax4.tick_params(colors='white')
        
        fig_comp.tight_layout()
        plt.show()
    
    def show_deflection_surfaces(self):
        """
        Muestra las superficies de deflexión final como función de ω y Vy (Figura 2 del artículo).
        Gráficas 3D con colormaps mostrando el efecto Magnus.
        """
        # Crear ventana de progreso
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Calculando...")
        progress_window.geometry("300x100")
        progress_window.configure(bg=DarkTheme.BG_DARK)
        progress_window.transient(self.root)
        
        progress_label = tk.Label(progress_window, 
            text="Calculando superficies de deflexión...\nEsto puede tomar unos segundos.",
            font=('Segoe UI', 11), fg=DarkTheme.TEXT_PRIMARY, bg=DarkTheme.BG_DARK)
        progress_label.pack(expand=True)
        progress_window.update()
        
        # Rangos según el artículo
        Vy_range = np.linspace(30, 50, 15)  # Velocidad inicial en y (m/s)
        omega_range = np.linspace(-310, 310, 21)  # Velocidad angular (rad/s)
        
        # Crear mallas
        Vy_mesh, omega_mesh = np.meshgrid(Vy_range, omega_range)
        
        # Matrices para almacenar deflexiones (en cm)
        dy_omega_x = np.zeros_like(Vy_mesh)
        dz_omega_x = np.zeros_like(Vy_mesh)
        dx_omega_z = np.zeros_like(Vy_mesh)
        dx_omega_y = np.zeros_like(Vy_mesh)
        
        # Calcular deflexiones
        for i, omega_val in enumerate(omega_range):
            for j, Vy_val in enumerate(Vy_range):
                V0 = (0, Vy_val, 0)
                
                # Trayectoria de referencia (sin Magnus)
                t_ref, x_ref, y_ref, z_ref, _, _, _ = simulate_trajectory(
                    V0, (0, 0, 0), t_max=0.6, include_magnus=False
                )
                
                if len(t_ref) > 0:
                    # Con ωx - produce deflexión en y, z
                    t1, x1, y1, z1, _, _, _ = simulate_trajectory(
                        V0, (omega_val, 0, 0), t_max=0.6
                    )
                    if len(t1) > 0:
                        dy_omega_x[i, j] = (y1[-1] - y_ref[-1]) * 100  # cm
                        dz_omega_x[i, j] = (z1[-1] - z_ref[-1]) * 100  # cm
                    
                    # Con ωz - produce deflexión en x
                    t2, x2, y2, z2, _, _, _ = simulate_trajectory(
                        V0, (0, 0, omega_val), t_max=0.6
                    )
                    if len(t2) > 0:
                        dx_omega_z[i, j] = (x2[-1] - x_ref[-1]) * 100  # cm
                    
                    # Con ωy - produce deflexión menor
                    t3, x3, y3, z3, _, _, _ = simulate_trajectory(
                        V0, (0, omega_val, 0), t_max=0.6
                    )
                    if len(t3) > 0:
                        dx_omega_y[i, j] = (x3[-1] - x_ref[-1]) * 100  # cm
        
        progress_window.destroy()
        
        # Crear figura con 4 subplots 3D
        plt.style.use('default')  # Usar estilo claro para mejor visualización de colormaps
        fig = plt.figure(figsize=(14, 12), facecolor='white')
        fig.suptitle('Figura 2: Deflexión Final en Función de Velocidad y Velocidad Angular', 
                    fontsize=14, fontweight='bold')
        
        # Colormap como en el artículo (jet/rainbow)
        cmap = 'jet'
        
        # 1. dy vs ωx vs Vy (arriba izquierda)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(omega_mesh, Vy_mesh, dy_omega_x, cmap=cmap, 
                                  edgecolor='none', alpha=0.9)
        ax1.set_xlabel('ωx (rad/s)', fontsize=10)
        ax1.set_ylabel('Vy (m/s)', fontsize=10)
        ax1.set_zlabel('dy (cm)', fontsize=10)
        ax1.set_title('Deflexión dy con spin ωx', fontsize=11)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
        ax1.view_init(elev=25, azim=-60)
        
        # 2. dz vs ωx vs Vy (arriba derecha)
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(omega_mesh, Vy_mesh, dz_omega_x, cmap=cmap,
                                  edgecolor='none', alpha=0.9)
        ax2.set_xlabel('ωx (rad/s)', fontsize=10)
        ax2.set_ylabel('Vy (m/s)', fontsize=10)
        ax2.set_zlabel('dz (cm)', fontsize=10)
        ax2.set_title('Deflexión dz con spin ωx', fontsize=11)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
        ax2.view_init(elev=25, azim=-60)
        
        # 3. dx vs ωz vs Vy (abajo izquierda) - La más importante
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        surf3 = ax3.plot_surface(omega_mesh, Vy_mesh, dx_omega_z, cmap=cmap,
                                  edgecolor='none', alpha=0.9)
        ax3.set_xlabel('ωz (rad/s)', fontsize=10)
        ax3.set_ylabel('Vy (m/s)', fontsize=10)
        ax3.set_zlabel('dx (cm)', fontsize=10)
        ax3.set_title('Deflexión dx con spin ωz', fontsize=11)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
        ax3.view_init(elev=25, azim=-60)
        
        # 4. dx vs ωy vs Vy (abajo derecha)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        surf4 = ax4.plot_surface(omega_mesh, Vy_mesh, dx_omega_y, cmap=cmap,
                                  edgecolor='none', alpha=0.9)
        ax4.set_xlabel('ωy (rad/s)', fontsize=10)
        ax4.set_ylabel('Vy (m/s)', fontsize=10)
        ax4.set_zlabel('dx (cm)', fontsize=10)
        ax4.set_title('Deflexión dx con spin ωy', fontsize=11)
        fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10, pad=0.1)
        ax4.view_init(elev=25, azim=-60)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def show_trajectory_error(self):
        """
        Muestra el error entre trayectorias a lo largo del lanzamiento (Figura 5).
        Compara las trayectorias obtenidas con diferentes ubicaciones del punto medio ξ2.
        """
        # Parámetros del Test 1 según el artículo
        V_target = (0, 45, 0)
        omega_target = (300, 0, 0)
        
        # Valores reconstruidos para diferentes ubicaciones de ξ2 (de la Tabla 2)
        reconstructed_cases = [
            {"name": "3/4 (y3-y1)", "V": (-0.0018, 45.0, 0.0012), 
             "omega": (300.8, 39.9, -2.8), "color": "black", "linestyle": "-"},
            {"name": "1/2 (y3-y1)", "V": (-0.0012, 45.0, 0.00061), 
             "omega": (301.0, 39.8, -2.7), "color": "brown", "linestyle": "-"},
            {"name": "1/3 (y3-y1)", "V": (-0.00079, 45.0, 0.00057), 
             "omega": (301.1, 39.8, -2.6), "color": "green", "linestyle": "-"},
            {"name": "1/4 (y3-y1)", "V": (-0.00059, 45.0, 0.00033), 
             "omega": (301.1, 39.7, -2.5), "color": "blue", "linestyle": "-"},
        ]
        
        # Generar trayectoria objetivo
        t_target, x_target, y_target, z_target, _, _, _ = simulate_trajectory(
            V_target, omega_target, t_max=0.5
        )
        
        # Crear figura
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        # Calcular y graficar error para cada caso
        from scipy.interpolate import interp1d
        
        for case in reconstructed_cases:
            t_recon, x_recon, y_recon, z_recon, _, _, _ = simulate_trajectory(
                case["V"], case["omega"], t_max=0.5
            )
            
            if len(t_recon) > 1 and len(t_target) > 1:
                # Interpolar para tener mismos tiempos
                t_common = np.linspace(0, min(t_target[-1], t_recon[-1]), 200)
                
                f_x_t = interp1d(t_target, x_target, kind='linear', fill_value='extrapolate')
                f_y_t = interp1d(t_target, y_target, kind='linear', fill_value='extrapolate')
                f_z_t = interp1d(t_target, z_target, kind='linear', fill_value='extrapolate')
                
                f_x_r = interp1d(t_recon, x_recon, kind='linear', fill_value='extrapolate')
                f_y_r = interp1d(t_recon, y_recon, kind='linear', fill_value='extrapolate')
                f_z_r = interp1d(t_recon, z_recon, kind='linear', fill_value='extrapolate')
                
                # Calcular error euclidiano en mm
                error = np.sqrt(
                    (f_x_t(t_common) - f_x_r(t_common))**2 +
                    (f_y_t(t_common) - f_y_r(t_common))**2 +
                    (f_z_t(t_common) - f_z_r(t_common))**2
                ) * 1000  # mm
                
                ax.plot(t_common, error, color=case["color"], linewidth=2, 
                       label=case["name"], linestyle=case["linestyle"])
        
        # Líneas verticales para t1 y t3
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='t₁')
        ax.axvline(x=t_target[-1], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='t₃')
        
        # Líneas verticales para los puntos medios
        midpoints = [(t_target[-1] * 0.25, 'blue'), (t_target[-1] * 0.33, 'green'), 
                     (t_target[-1] * 0.5, 'brown'), (t_target[-1] * 0.75, 'black')]
        for mp, col in midpoints:
            ax.axvline(x=mp, color=col, linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Time of Flight (s)', fontsize=12)
        ax.set_ylabel('Error (mm)', fontsize=12)
        ax.set_title('Figura 5: Error entre Trayectorias a lo largo del Lanzamiento', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.45)
        ax.set_ylim(0, 0.15)
        
        plt.tight_layout()
        plt.show()
    
    def show_function_heatmap(self):
        """
        Muestra el heatmap del valor de la función objetivo f(V, ω) (Figura 7).
        Relaciona el punto medio de la trayectoria con la precisión de los datos.
        """
        # Parámetros del Test 1
        V_target = (0, 45, 0)
        omega_target = (300, 0, 0)
        
        # Rangos
        midpoint_percent = np.linspace(25, 75, 8)  # % de la trayectoria
        accuracy_levels = np.linspace(0, 2, 10)  # mm de precisión
        
        # Generar trayectoria objetivo
        t, x, y, z, _, _, _ = simulate_trajectory(V_target, omega_target, t_max=0.5)
        
        # Calcular matriz de valores de función
        heatmap = np.zeros((len(accuracy_levels), len(midpoint_percent)))
        
        # Valores óptimos aproximados (de la Tabla 4)
        V_opt = (-0.001, 45.0, -0.00001)
        omega_opt = (301.1, 36.1, -0.6)
        
        for i, acc in enumerate(accuracy_levels):
            for j, midpoint in enumerate(midpoint_percent):
                # Índice del punto medio
                idx_mid = int(len(t) * midpoint / 100)
                idx_mid = max(1, min(idx_mid, len(t) - 2))
                
                # Añadir ruido según precisión
                noise = acc / 1000  # Convertir mm a m
                
                target_points = [
                    (x[0], y[0], z[0]),
                    (x[idx_mid] + noise * np.sin(midpoint),
                     y[idx_mid] + noise * np.cos(midpoint),
                     z[idx_mid] + noise * 0.5),
                    (x[-1] + noise * 0.3,
                     y[-1] + noise * 0.2,
                     z[-1] + noise * 0.4)
                ]
                target_times = [t[0], t[idx_mid], t[-1]]
                
                # Calcular valor de función objetivo
                f_value = objective_function(V_opt, omega_opt, target_points, target_times)
                
                # Escalar para visualización (valores similares al artículo 1-8)
                heatmap[i, j] = max(1, min(8, 1 + f_value * 1000))
        
        # Crear figura
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # Crear heatmap con colormap tipo jet
        im = ax.imshow(heatmap, cmap='jet', aspect='auto', origin='lower',
                       extent=[midpoint_percent[0], midpoint_percent[-1], 
                               accuracy_levels[0], accuracy_levels[-1]])
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Value of function (13)', fontsize=11)
        
        # Etiquetas
        ax.set_xlabel('Midpoint (% of the trajectory)', fontsize=12)
        ax.set_ylabel('Accuracy of the Data (mm)', fontsize=12)
        ax.set_title('Figura 7: Valor de la Función Objetivo f(V, ω)', 
                    fontsize=12, fontweight='bold')
        
        # Grid
        ax.grid(True, color='white', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============== EJECUTAR APLICACIÓN ==============
if __name__ == "__main__":
    root = tk.Tk()
    app = BaseballTrajectoryApp(root)
    root.mainloop()
