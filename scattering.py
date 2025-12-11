import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# --- 1. 시스템 매개변수 설정 ---
L = 50       # 파동 패킷의 길이 (및 대략적인 최대 시뮬레이션 시간)
N_graph = 300  # 그래프의 절반 크기 (x: -300 ~ 300)
N_total = 2 * N_graph + 1
dt = 0.5     # 시간 간격
k_center = np.pi / 2  # NAND Gate 작동을 위한 중심 파수 (lambda=0)

# X 좌표 축 (시각화용)
x_coords = np.arange(-N_graph, N_graph + 1)

# --- 2. 해밀토니안 구성 함수 ---


def build_hamiltonian(alpha_prime):
    """
    등가 자기 루프 alpha_prime을 포함한 교란된 라인 그래프의 해밀토니안을 구성합니다.
    """
    H = np.zeros((N_total, N_total), dtype=float)

    # H_0: 이웃 간 연결 (1/2)
    for i in range(N_total):
        if i > 0:
            H[i, i-1] = 0.5
        if i < N_total - 1:
            H[i, i+1] = 0.5

    # 교란: 원점 (N_graph)에 alpha' 적용
    H[N_graph, N_graph] = alpha_prime
    return H

# --- 3. 초기 파동 패킷 구성 함수 ---


def initialize_wave_packet(L_val, k_val, N_graph_val):
    """
    길이 L, 중심 파수 k를 가진 초기 파동 패킷 Psi_0를 구성합니다.
    """
    Psi_0 = np.zeros(N_total, dtype=complex)

    # 패킷은 x = -L 에서 x = 0 (index N_graph)까지 정의됩니다.
    start_idx = N_graph - L_val
    end_idx = N_graph

    x_indices = np.arange(start_idx, end_idx) - N_graph_val

    # Psi(x) = (1/sqrt(L)) * exp(i * k * x)
    Psi_0[start_idx:end_idx] = (1 / np.sqrt(L_val)) * \
        np.exp(1j * k_val * x_indices)
    return Psi_0

# --- 4. 시뮬레이션 실행 및 플롯 ---


def run_and_plot(alpha_prime, title, plot_times):
    H = build_hamiltonian(alpha_prime)
    Psi_t = initialize_wave_packet(L, k_center, N_graph)

    # 시간 진화 연산자 계산 (상태는 복소수여야 합니다)
    # H는 실수 행렬이지만, 지수 연산 expm(1j * H * dt)를 위해 복소수 계산을 사용합니다.
    U_dt = expm(1j * H * dt)

    fig, axes = plt.subplots(
        len(plot_times), 1, figsize=(10, 2 * len(plot_times)))
    fig.suptitle(
        f'Wave Packet Scattering: {title} ($\\alpha\' = {alpha_prime}$)', fontsize=14)

    current_time = 0.0

    for t_step in range(int(L / dt) + 1):
        if current_time in plot_times:
            idx = plot_times.index(current_time)
            prob_density = np.abs(Psi_t)**2

            axes[idx].plot(x_coords, prob_density, 'b-')
            axes[idx].set_title(f't = {current_time:.1f}', loc='left')
            # Origin (Tree Root)
            axes[idx].axvline(x=0, color='r', linestyle='--')
            axes[idx].set_xlim(-L - 10, L + 10)
            axes[idx].set_ylim(0, 0.04)  # 확률 밀도 최대값 설정

            # 반사/투과 영역 강조
            axes[idx].axvspan(0, L + 10, color='g', alpha=0.1)  # 투과 영역
            axes[idx].axvspan(-L - 10, 0, color='y', alpha=0.1)  # 반사 영역

        # 상태 업데이트: Psi_t+dt = U_dt * Psi_t
        Psi_t = U_dt @ Psi_t
        current_time += dt

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 5. 시뮬레이션 실행 (True vs. False) ---


# 시뮬레이션 시간 스냅샷 (PDF Figure 15.18과 유사)
PLOT_TIMES = [0.0, 25.0, 50.0, 75.0]

# Scenario 1: NAND Tree is TRUE (반사)
# True: alpha' = Inf (매우 큰 값으로 근사, 완전 반사체)
ALPHA_TRUE = 100.0
run_and_plot(ALPHA_TRUE, "TRUE (Reflected)", PLOT_TIMES)

# Scenario 2: NAND Tree is FALSE (투과)
# False: alpha' = 0 (투명, 완전 투과)
ALPHA_FALSE = 0.0
run_and_plot(ALPHA_FALSE, "FALSE (Transmitted)", PLOT_TIMES)
