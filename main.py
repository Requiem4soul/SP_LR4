import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings

warnings.filterwarnings('ignore')


# Задача 1: Аналитическое исследование

def analytical_ergodicity_test():
    """
    Аналитическое исследование процесса X(t) на эргодичность по математическому ожиданию
    """

    # Численная проверка для различных значений T
    T_values = np.logspace(0, 3, 50)  # от 1 до 1000
    criterion_values = []

    for T in T_values:
        integral_value = T + 2 * np.exp(-T) - 1
        criterion = integral_value / (T ** 2)
        criterion_values.append(criterion)

    plt.figure(figsize=(10, 6))
    plt.semilogx(T_values, criterion_values, 'b-', linewidth=2)
    plt.xlabel('T')
    plt.ylabel('Критерий эргодичности')
    plt.title('Сходимость критерия эргодичности к нулю')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.show()


# Задача 2: Эмпирическое исследование модели Спарре-Андерсона

class SparreAndersenModel:
    """
    Модель страхования Спарре-Андерсона
    - u: начальный капитал
    - c: премии в единицу времени  
    - S(t): агрегированный процесс выплат

    - N(t): процесс подсчета страховых случаев (процесс Пуассона)
    - X_i: размеры отдельных выплат (независимые одинаково распределенные)
    """

    def __init__(self, initial_capital=10000, premium_rate=200, claim_rate=2,
                 claim_mean=400, claim_std=2, seed=None):
        """
        Параметры модели:
        - initial_capital: u
        - premium_rate:  c
        - claim_rate:  λ (для процесса Пуассона)
        - claim_mean: среднее значение выплаты
        - claim_std: стандартное отклонение выплаты
        """
        self.u = initial_capital
        self.c = premium_rate
        self.lam = claim_rate
        self.claim_mean = claim_mean
        self.claim_std = claim_std

        if seed is not None:
            np.random.seed(seed)

    def simulate_path(self, T, dt=0.1):
        """
        Симулирует одну траекторию процесса U(t) на интервале [0,T]

        Параметры:
        - T: конечное время
        - dt: шаг дискретизации

        Возвращает:
        - t_grid: сетка времени
        - U_path: значения процесса U(t)
        """
        t_grid = np.arange(0, T + dt, dt)
        n_steps = len(t_grid)
        U_path = np.zeros(n_steps)
        U_path[0] = self.u

        # Симулируем процесс пошагово
        for i in range(1, n_steps):
            # Количество новых страховых случаев за время dt
            # (Пуассоновский процесс с интенсивностью λ)
            n_claims = np.random.poisson(self.lam * dt)

            # Общая сумма выплат за время dt
            if n_claims > 0:
                # Выплаты распределены нормально (обрезаем отрицательные значения)
                claims = np.maximum(0, np.random.normal(self.claim_mean, self.claim_std, n_claims))
                total_claims = np.sum(claims)
            else:
                total_claims = 0

            # Обновляем капитал: U(t+dt) = U(t) + c*dt - выплаты
            U_path[i] = U_path[i - 1] + self.c * dt - total_claims

        return t_grid, U_path

    def simulate_multiple_paths(self, T, n_paths, dt=0.1):
        """
        Симулирует множество независимых траекторий
        """
        all_paths = []
        t_grid = None

        for _ in range(n_paths):
            t, U = self.simulate_path(T, dt)
            all_paths.append(U)
            if t_grid is None:
                t_grid = t

        return t_grid, np.array(all_paths)


def empirical_ergodicity_test():
    """
    Эмпирическое исследование модели Спарре-Андерсона на эргодичность
    """
    print()

    # Создаем модель с фиксированными параметрами
    model = SparreAndersenModel(
        initial_capital=10000,
        premium_rate=3500,  # c = 200
        claim_rate=2,  # λ = 2 (в среднем 2 страховых случая в единицу времени)
        claim_mean=1500,  # E[X_i] = 4
        claim_std=200,  # σ[X_i] = 2
        seed=42
    )

    print("Параметры модели:")
    print(f"- Начальный капитал: {model.u}")
    print(f"- Интенсивность премий: {model.c}")
    print(f"- Интенсивность страховых случаев: {model.lam}")
    print(f"- Среднее значение выплаты: {model.claim_mean}")
    print(f"- Стандартное отклонение выплаты: {model.claim_std}")
    print()

    # Теоретическое математическое ожидание процесса
    # E[U(t)] = u + (c - λ*E[X])t
    theoretical_drift = model.c - model.lam * model.claim_mean
    print(f"Теоретический дрейф: c - λ*E[X] = {model.c} - {model.lam}*{model.claim_mean} = {theoretical_drift}")

    if theoretical_drift > 0:
        print("Дрейф положительный → капитал растет в среднем")
    elif theoretical_drift < 0:
        print("Дрейф отрицательный → капитал убывает в среднем (возможно разорение)")
    else:
        print("Дрейф нулевой → капитал стационарен в среднем")
    print()

    # Симулируем множество траекторий для разных временных горизонтов
    T_values = [50, 100, 200, 500]
    n_paths = 1000

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    ergodicity_results = []

    for idx, T in enumerate(T_values):
        print(f"Анализ для T = {T}:")

        # Симулируем траектории
        t_grid, paths = model.simulate_multiple_paths(T, n_paths, dt=0.5)

        # Вычисляем временные средние для каждой траектории
        time_averages = []
        for path in paths:
            # Временное среднее: среднее значение по траектории
            time_avg = np.mean(path)
            time_averages.append(time_avg)

        time_averages = np.array(time_averages)

        # Вычисляем ансамблевое среднее (среднее по всем траекториям в каждый момент времени)
        ensemble_means = np.mean(paths, axis=0)
        final_ensemble_mean = ensemble_means[-1]  # среднее в конце периода

        # Статистики временных средних
        mean_time_avg = np.mean(time_averages)
        std_time_avg = np.std(time_averages)

        print(f"  Среднее временных средних: {mean_time_avg:.2f}")
        print(f"  Стд. откл. временных средних: {std_time_avg:.2f}")
        print(f"  Ансамблевое среднее в конце: {final_ensemble_mean:.2f}")
        print(f"  Разность |временное - ансамблевое|: {abs(mean_time_avg - final_ensemble_mean):.2f}")

        # Критерий эргодичности: временные средние должны быть близки к ансамблевому среднему
        ergodicity_measure = std_time_avg / abs(mean_time_avg) if mean_time_avg != 0 else float('inf')
        ergodicity_results.append(ergodicity_measure)

        print(f"  Мера эргодичности (CV): {ergodicity_measure:.4f}")
        print()

        # Визуализация
        ax = axes[idx]

        # Показываем несколько примеров траекторий
        for i in range(min(10, n_paths)):
            ax.plot(t_grid, paths[i], 'b-', alpha=0.3, linewidth=0.5)

        # Показываем ансамблевое среднее
        ax.plot(t_grid, ensemble_means, 'r-', linewidth=2, label='Ансамблевое среднее')

        # Показываем теоретическое среднее
        theoretical_mean = model.u + theoretical_drift * t_grid
        ax.plot(t_grid, theoretical_mean, 'g--', linewidth=2, label='Теоретическое среднее')

        ax.set_xlabel('Время t')
        ax.set_ylabel('Капитал U(t)')
        ax.set_title(f'T = {T}, CV = {ergodicity_measure:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    print("Выводы по эмпирическому исследованию")
    print(f"Коэффициенты вариации: {[f'{x:.4f}' for x in ergodicity_results]}")

    if len(ergodicity_results) > 1 and ergodicity_results[-1] < ergodicity_results[0]:
        print("Наблюдается улучшение эргодичности с увеличением T")

    return ergodicity_results


# ===== ОСНОВНАЯ ФУНКЦИЯ =====

def main():
    """
    Главная функция для выполнения всех заданий
    """
    print("")

    # Задача 1: Аналитическое исследование
    analytical_ergodicity_test()

    print("")

    # Задача 2: Эмпирическое исследование
    ergodicity_results = empirical_ergodicity_test()

    print("")
    print(f"Финальный коэффициент вариации: {ergodicity_results[-1]:.4f}")


if __name__ == "__main__":
    main()