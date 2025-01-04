package ltd.mbor.sciko.ml.linear

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

class LinearRegression {
  var weights: MultiArray<Double, D1>? = null
  var bias: Double = 0.0

  fun fit(X: MultiArray<Double, D2>, y: MultiArray<Double, D1>): LinearRegression {
    val ones = mk.ones<Double>(X.shape[0], 1)
    val X_b = mk.hstack(ones, X)
    val pseudoInverse = mk.linalg.inv(X_b.transpose().dot(X_b)).dot(X_b.transpose())
    val weightsWithBias = pseudoInverse.dot(y)
    bias = weightsWithBias[0]
    weights = weightsWithBias[1..<weightsWithBias.size]
    return this
  }

  fun fit(X: DataFrame, y: DataFrame.Column<Double>): LinearRegression {
    return fit(mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose(), mk.ndarray(y.values))
  }

  fun predict(X: MultiArray<Double, D2>): MultiArray<Double, D1> {
    val ones = mk.ones<Double>(X.shape[0], 1)
    val X_b = mk.hstack(ones, X)
    return X_b.dot(mk.ndarray(listOf(bias) + checkNotNull(weights).toList()))
  }

  fun predict(X: DataFrame): MultiArray<Double, D1> {
    return predict(mk.ndarray(X.columns.map { (it as DataFrame.Column<Double>).values }).transpose())
  }

  fun predict(x: MultiArray<Double, D1>): Double {
    val ones = mk.ones<Double>(1)
    val x_b = ones cat x
    return x_b.dot(mk.ndarray(listOf(bias) + checkNotNull(weights).toList()))
  }
}

private fun Multik.hstack(a: MultiArray<Double, D2>, b: MultiArray<Double, D2>): MultiArray<Double, D2> {
  return a.cat(b, axis = 1)
}