"""Собственные реализации линейных моделей для лабораторной работы №3."""

import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучить линейную регрессию пакетным градиентным спуском.

    Признаки рекомендуется предварительно масштабировать. Функция принимает
    уже подготовленные числовые данные и возвращает веса, смещение, историю
    MSE на обучающей выборке и прогнозы для тестовой выборки.
    """
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Признаки должны быть двумерными массивами.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Количество объектов и целевых значений не совпадает.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("У train и test должно быть одинаковое число признаков.")
    if x_train.shape[0] == 0 or x_train.shape[1] == 0:
        raise ValueError("Обучающая выборка не должна быть пустой.")
    if not (np.isfinite(x_train).all() and np.isfinite(x_test).all()
            and np.isfinite(y_train).all()):
        raise ValueError("В данных обнаружены NaN или бесконечные значения.")
    if data.learning_rate <= 0 or data.epochs <= 0:
        raise ValueError("learning_rate и epochs должны быть положительными.")

    n_objects, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    # Пакетный градиентный спуск: на каждой эпохе используются все объекты.
    for _ in range(data.epochs):
        predictions = x_train @ weights + bias
        errors = predictions - y_train

        grad_weights = (2.0 / n_objects) * (x_train.T @ errors)
        grad_bias = 2.0 * np.mean(errors)

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

        # Сохраняем MSE после обновления параметров на текущей эпохе.
        updated_predictions = x_train @ weights + bias
        mse = np.mean((updated_predictions - y_train) ** 2)
        loss_history.append(float(mse))

    test_predictions = x_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучить собственный бинарный классификатор пакетным градиентным спуском.

    Функция получает числовые и предварительно подготовленные признаки.
    В качестве функции потерь используется бинарная кросс-энтропия.
    """
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Признаки должны быть двумерными массивами.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Количество объектов и целевых значений не совпадает.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("У train и test должно быть одинаковое число признаков.")
    if x_train.shape[0] == 0 or x_train.shape[1] == 0:
        raise ValueError("Обучающая выборка не должна быть пустой.")
    if not (np.isfinite(x_train).all() and np.isfinite(x_test).all()
            and np.isfinite(y_train).all()):
        raise ValueError("В данных обнаружены NaN или бесконечные значения.")
    if not np.isin(y_train, [0.0, 1.0]).all():
        raise ValueError("Целевая переменная должна содержать только классы 0 и 1.")
    if data.learning_rate <= 0 or data.epochs <= 0:
        raise ValueError("learning_rate и epochs должны быть положительными.")

    n_objects, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    def sigmoid(logits: np.ndarray) -> np.ndarray:
        # Ограничение аргумента защищает exp от переполнения.
        logits = np.clip(logits, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-logits))

    for _ in range(data.epochs):
        logits = x_train @ weights + bias
        probabilities = sigmoid(logits)
        errors = probabilities - y_train

        # Градиенты бинарной кросс-энтропии по всем объектам train.
        grad_weights = (x_train.T @ errors) / n_objects
        grad_bias = float(np.mean(errors))

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

        # logaddexp вычисляет BCE устойчиво даже при больших значениях logits.
        updated_logits = x_train @ weights + bias
        loss = np.mean(np.logaddexp(0.0, updated_logits) - y_train * updated_logits)
        loss_history.append(float(loss))

    test_probabilities = sigmoid(x_test @ weights + bias)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравнить sklearn Logistic Regression, Decision Tree и KNN на одном split.

    Собственная логистическая регрессия обучается отдельно функцией
    train_logistic_classifier; её метрики сохраняются в общем experiment_results.json.
    Здесь реализованы три классических базовых алгоритма, предусмотренные
    BaselineComparisonResult и контрактом стартового файла.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    x_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Признаки должны быть двумерными массивами.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Количество объектов train и целевых значений не совпадает.")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("Количество объектов test и целевых значений не совпадает.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("У train и test должно быть одинаковое число признаков.")
    if not (np.isfinite(x_train).all() and np.isfinite(x_test).all()):
        raise ValueError("В признаках обнаружены NaN или бесконечные значения.")
    if not (np.isin(y_train, [0, 1]).all() and np.isin(y_test, [0, 1]).all()):
        raise ValueError("Целевая переменная должна содержать только классы 0 и 1.")
    if np.unique(y_train).size != 2:
        raise ValueError("В обучающей выборке должны присутствовать оба класса.")

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    def calculate_metrics(model) -> ClassificationMetrics:
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = model.predict(x_test)
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(y_test, predictions, zero_division=0)),
            recall=float(recall_score(y_test, predictions, zero_division=0)),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
        )

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = calculate_metrics(model)

    return BaselineComparisonResult(
        logistic_regression=results["logistic_regression"],
        decision_tree=results["decision_tree"],
        knn=results["knn"],
    )
