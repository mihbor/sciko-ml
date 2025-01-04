package ltd.mbor.sciko.ml.metrics

import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import kotlin.math.abs

fun meanSquaredError(yTrue: MultiArray<Double, D1>, yPred: MultiArray<Double, D1>): Double {
  val errors = yTrue - yPred
  return (errors dot errors) / yTrue.size
}

fun meanAbsoluteError(yTrue: MultiArray<Double, D1>, yPred: MultiArray<Double, D1>): Double {
  val errors = yTrue - yPred
  return errors.map { abs(it) }.sum() / yTrue.size
}

fun r2Score(yTrue: MultiArray<Double, D1>, yPred: MultiArray<Double, D1>): Double {
  val yTrueMean = yTrue.sum() / yTrue.size
  val ssTot = (yTrue - yTrueMean) dot (yTrue - yTrueMean)
  val ssRes = (yTrue - yPred) dot (yTrue - yPred)
  return 1 - ssRes / ssTot
}
