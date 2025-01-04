package ltd.mbor.sciko.ml

import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

fun assertEquals(expected: MultiArray<Double, D1>, actual: MultiArray<Double, D1>?, absoluteTolerance: Double) {
  for (i in expected.indices) {
    kotlin.test.assertEquals(expected[i], actual!![i], absoluteTolerance)
  }
}

fun assertEquals(expected: List<Double>, actual: List<Double>?, absoluteTolerance: Double) {
  for (i in expected.indices) {
    kotlin.test.assertEquals(expected[i], actual!![i], absoluteTolerance)
  }
}