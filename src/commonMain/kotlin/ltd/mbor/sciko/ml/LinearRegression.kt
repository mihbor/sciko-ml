package ltd.mbor.sciko.ml

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import org.jetbrains.kotlinx.multik.ndarray.linalg.*

class LinearRegression {
    private lateinit var weights: MultiArray<Double, D1>
    private var bias: Double = 0.0

    fun fit(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>): LinearRegression {
        val ones = mk.ones<Double>(X.shape[0], 1)
        val X_b = mk.hstack(ones, X)
        val pseudoInverse = mk.linalg.inv(X_b.transpose().dot(X_b)).dot(X_b.transpose())
        val weightsWithBias = pseudoInverse.dot(y)
        bias = weightsWithBias[0]
        weights = weightsWithBias[1..weightsWithBias.size]
        return this
    }

    fun predict(X: MultiArray<Double, D2>): MultiArray<Double, D1> {
        val ones = mk.ones<Double>(X.shape[0], 1)
        val X_b = mk.hstack(ones, X)
        return X_b.dot(mk.ndarray(listOf(bias) + weights.toList()))
    }
}
