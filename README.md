# ⚾ Three-Point-Based Algorithm - Baseball Trajectory Simulator

**Desarrollado por: Juan Diego Sarabia Cadenas**

Simulador interactivo de trayectorias de béisbol basado en el algoritmo de tres puntos para la reconstrucción de lanzamientos con spin.

## 📥 Cómo Clonar el Repositorio

```bash
git clone https://github.com/sarabiapillo/three-point-based_algorithm_v1.git
cd three-point-based_algorithm_v1
```

**Nota:** Si te pide contraseña o credenciales, usa esta alternativa:
```bash
# Opción 1: Descargar como ZIP
# Ve a https://github.com/sarabiapillo/three-point-based_algorithm_v1
# Haz clic en "Code" > "Download ZIP"

# Opción 2: Clonar con HTTPS (sin autenticación para repos públicos)
git clone https://github.com/sarabiapillo/three-point-based_algorithm_v1.git
```

## 📋 Descripción

Esta aplicación implementa las ecuaciones físicas reales del movimiento de una pelota de béisbol, incluyendo:

- **Fuerza de Arrastre (Drag)** - Resistencia del aire
- **Fuerza de Magnus** - Efecto del spin en la trayectoria  
- **Fuerza Gravitacional** - Aceleración por gravedad
- **Fuerza Centrífuga** - Efecto de la rotación terrestre

### 🎯 Propósito

El objetivo de esta aplicación es **verificar experimentalmente** los cálculos y resultados presentados en la investigación científica sobre reconstrucción de trayectorias de lanzamientos de béisbol. 

**La aplicación NO inventa datos** - todos los cálculos se basan en:
- Ecuaciones de movimiento de Newton
- Coeficientes aerodinámicos experimentales (Cd, Cm)
- Parámetros físicos reales (masa, diámetro, densidad del aire)
- Métodos numéricos Runge-Kutta de 4to orden

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip (gestor de paquetes)

### Dependencias
```bash
pip install numpy scipy matplotlib
```

### Ejecutar la aplicación
```bash
python baseball_trajectory_app.py
```

## 📊 Características

### Simulación de Trayectorias
- Visualización 3D interactiva
- Proyecciones 2D (vista superior, frontal y lateral)
- Zona de strike visual
- Cálculo de deflexiones en tiempo real

### Parámetros Ajustables
| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| Vx | -5 a 5 m/s | Velocidad lateral |
| Vy | 20 a 55 m/s | Velocidad hacia el home |
| Vz | -5 a 5 m/s | Velocidad vertical |
| ωx, ωy, ωz | -310 a 310 rad/s | Componentes del spin |

### Casos de Prueba Incluidos
1. **Fastball** - V=(0, 45, 0) m/s, ω=(300, 0, 0) rad/s
2. **Curveball** - V=(2, 39, 1) m/s, ω=(30, 80, 200) rad/s
3. **Slider** - V=(1, 42, 0.5) m/s, ω=(150, 100, 150) rad/s

### Gráficas del Artículo
- **Figura 2**: Superficies 3D de deflexión (dx, dy, dz vs ω vs Vy)
- **Figura 5**: Error entre trayectorias a lo largo del lanzamiento
- **Figura 7**: Heatmap del valor de la función objetivo f(V, ω)

### Algoritmo de Reconstrucción
Implementación del método de tres puntos:
- Minimización de f(V, ω) = ||ξ₂ - p₂|| + ||ξ₃ - p₃||
- Método de shooting con Newton-Raphson (velocidad)
- Método de la secante (velocidad angular)

## 📐 Ecuaciones Implementadas

### Ecuación de Movimiento (Ec. 7)
```
dVx/dt = (k/mV)[Cm/ω(ωyVz - ωzVy) - CdVx] + Ω²R·sin(ψ)cos(ψ)sin(γ)
dVy/dt = (k/mV)[Cm/ω(ωzVx - ωxVz) - CdVy] + Ω²R·sin(ψ)cos(ψ)cos(γ)
dVz/dt = (k/mV)[Cm/ω(ωxVy - ωyVx) - CdVz] + Ω²R·sin²(ψ) - g
```

### Coeficiente de Arrastre (Ec. 11)
```
Cd(V) = 0.29 + 0.22 / (1 + exp((V - 32.37) / 5.2) - 1)
```

### Coeficiente de Magnus (Ec. 4)
```
Cm(ω) = 0.319 × (1 - exp(-2.48×10⁻³ × ω))
```

## 🔬 Constantes Físicas Utilizadas

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| Masa | 142 g | Masa oficial de la pelota |
| Diámetro | 7.16 cm | Diámetro oficial |
| ρ (aire) | 1.22 kg/m³ | Densidad del aire |
| g | 9.8 m/s² | Aceleración gravitacional |

## 📸 Capturas de Pantalla

La aplicación incluye:
- Panel de control con sliders interactivos
- Visualización de trayectoria 3D
- 4 vistas simultáneas (3D, superior, frontal, lateral)
- Tema oscuro moderno
- Información en tiempo real de coeficientes y deflexiones

## 🧪 Validación

Los usuarios pueden verificar que:
1. Las deflexiones calculadas coinciden con los valores teóricos
2. El algoritmo de reconstrucción converge a los parámetros originales
3. Las gráficas reproducen los resultados publicados en la literatura

## 📁 Estructura del Proyecto

```
three-point-based_algorithm_v1/
├── baseball_trajectory_app.py   # Aplicación principal
├── README.md                    # Este archivo
├── paper_baseball1.pdf          # Artículo de referencia
├── documento_traducido.tex      # Documento LaTeX traducido
├── texto_original.txt           # Texto extraído del PDF
├── texto_traducido.txt          # Texto traducido al español
└── extracted_images/            # Imágenes del artículo
    ├── imagen_1_1.jpeg
    ├── imagen_1_2.jpeg
    └── ...
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Abre un Pull Request

## 📄 Licencia

MIT License - Copyright (c) 2026 Juan Diego Sarabia Cadenas

Este proyecto es de código abierto. Puedes usar, modificar y distribuir libremente.
Ver archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Juan Diego Sarabia Cadenas**
- GitHub: [@sarabiapillo](https://github.com/sarabiapillo)

## 📚 Referencias

Basado en metodologías de reconstrucción de trayectorias de béisbol utilizando:
- Dinámica de cuerpos en rotación
- Efecto Magnus en esferas
- Métodos numéricos para problemas de valor frontera

---

**Nota**: Esta aplicación es una herramienta de visualización y verificación. Los resultados deben interpretarse en el contexto de las aproximaciones y simplificaciones del modelo físico utilizado.
