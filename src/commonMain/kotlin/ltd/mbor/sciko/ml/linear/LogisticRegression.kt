package ltd.mbor.sciko.ml.linear

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.exp
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

class LogisticRegression {
    var weights: MultiArray<Double, D1>? = null
    var bias: Double = 0.0
    private val learningRate = 0.01
    private val numIterations = 1000

    fun fit(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>): LogisticRegression {
        val nSamples = X.shape[0]
        val nFeatures = X.shape[1]
        weights = mk.zeros(nFeatures)
        bias = 0.0

        for (i in 0 until numIterations) {
            val linearModel = X.dot(weights!!) + bias
            val yPredicted = sigmoid(linearModel)

            val dw = (1.0 / nSamples) * (X.transpose().dot(yPredicted - y))
            val db = (1.0 / nSamples) * (yPredicted - y).sum()

            weights = weights!! - learningRate * dw
            bias -= learningRate * db
        }

        return this
    }

    fun fit(X: DataFrame, y: DataFrame.Column<Double>): LogisticRegression {
        return fit(mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose(), mk.ndarray(y.values))
    }

    fun predict(X: MultiArray<Double, D2>): MultiArray<Double, D1> {
        val linearModel = X.dot(weights!!) + bias
        return sigmoid(linearModel).map { if (it >= 0.5) 1.0 else 0.0 }
    }

    fun predict(X: DataFrame): MultiArray<Double, D1> {
        return predict(mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose())
    }

    fun predictProbability(X: MultiArray<Double, D2>): MultiArray<Double, D1> {
        val linearModel = X.dot(weights!!) + bias
        return sigmoid(linearModel)
    }

    fun predictProbability(X: DataFrame): MultiArray<Double, D1> {
        return predictProbability(mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose())
    }

    private fun sigmoid(z: MultiArray<Double, D1>): MultiArray<Double, D1> {
        return 1.0 / (1.0 + exp(-z))
    }
}
