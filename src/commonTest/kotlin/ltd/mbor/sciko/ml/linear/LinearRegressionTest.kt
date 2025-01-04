package ltd.mbor.sciko.ml.linear

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class LinearRegressionTest {

  @Test
  fun testFit() {
    val X = mk.ndarray(mk[
      mk[1.0, 2.0],
      mk[2.0, 4.0],
      mk[3.0, 8.0]
    ])
    val y = mk.ndarray(mk[4.0, 7.0, 12.0])

    val model = LinearRegression().fit(X, y)

    assertEquals(1.0, model.bias, 1e-6)
    assertEquals(mk.ndarray(mk[1.0, 1.0]), model.weights, 1e-6)
  }

  @Test
  fun testFitDataFrame() {
    val csvData = """
        a,b,c
        1,2,4
        2,4,7
        3,8,12
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = LinearRegression().fit(X, y)

    assertEquals(1.0, model.bias, 1e-6)
    assertEquals(mk.ndarray(mk[1.0, 1.0]), model.weights, 1e-6)
  }

  @Test
  fun testPredict() {
    val X = mk.ndarray(mk[
      mk[1.0, 2.0],
      mk[2.0, 4.0],
      mk[3.0, 8.0]
    ])
    val y = mk.ndarray(mk[4.0, 7.0, 12.0])

    val model = LinearRegression().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y, predictions, 1e-6)
  }

  @Test
  fun testPredictDataFrame() {
    val csvData = """
        a,b,c
        1,2,4
        2,4,7
        3,8,12
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = LinearRegression().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y.values, predictions.toList(), 1e-6)
  }

  @Test
  fun testPredictOne() {
    val X = mk.ndarray(mk[
      mk[1.0, 2.0],
      mk[2.0, 4.0],
      mk[3.0, 8.0]
    ])
    val y = mk.ndarray(mk[4.0, 7.0, 12.0])

    val model = LinearRegression().fit(X, y)
    val prediction = model.predict(X[1])

    assertEquals(y[1], prediction, 1e-6)
  }
}

private fun assertEquals(expected: MultiArray<Double, D1>, actual: MultiArray<Double, D1>?, absoluteTolerance: Double) {
  for (i in expected.indices) {
    assertEquals(expected[i], actual!![i], absoluteTolerance)
  }
}

private fun assertEquals(expected: List<Double>, actual: List<Double>?, absoluteTolerance: Double) {
  for (i in expected.indices) {
    assertEquals(expected[i], actual!![i], absoluteTolerance)
  }
}