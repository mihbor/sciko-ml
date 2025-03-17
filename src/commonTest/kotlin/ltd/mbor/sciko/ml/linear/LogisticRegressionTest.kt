package ltd.mbor.sciko.ml.linear

import ltd.mbor.sciko.ml.assertEquals
import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class LogisticRegressionTest {

  @Test
  fun testFit() {
    val X = mk.ndarray(
      mk[
        mk[1.0, 2.0],
        mk[2.0, 4.0],
        mk[3.0, 8.0]
      ]
    )
    val y = mk.ndarray(mk[0.0, 1.0, 1.0])

    val model = LogisticRegression().fit(X, y)

    assertEquals(-1.034517, model.bias, 1e-6)
    assertEquals(mk.ndarray(mk[0.0810849, 0.532951]), model.weights, 1e-6)
  }

  @Test
  fun testFitDataFrame() {
    val csvData = """
      a,b,c
      1,2,0
      2,4,1
      3,8,1
      """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = LogisticRegression().fit(X, y)

    assertEquals(-1.034517, model.bias, 1e-6)
    assertEquals(mk.ndarray(mk[0.0810849, 0.532951]), model.weights, 1e-6)
  }

  @Test
  fun testPredict() {
    val X = mk.ndarray(
      mk[
        mk[1.0, 2.0],
        mk[2.0, 4.0],
        mk[3.0, 8.0]
      ]
    )
    val y = mk.ndarray(mk[0.0, 1.0, 1.0])

    val model = LogisticRegression().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y, predictions, 1e-6)
  }

  @Test
  fun testPredictDataFrame() {
    val csvData = """
      a,b,c
      1,2,0
      2,4,1
      3,8,1
      """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = LogisticRegression().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y.values, predictions.toList(), 1e-6)
  }

  @Test
  fun testPredictProbability() {
    val X = mk.ndarray(
      mk[
        mk[1.0, 2.0],
        mk[2.0, 4.0],
        mk[3.0, 8.0]
      ]
    )
    val y = mk.ndarray(mk[0.0, 1.0, 1.0])

    val model = LogisticRegression().fit(X, y)
    val probabilities = model.predictProbability(X)

    assertEquals(mk.ndarray(mk[0.5, 0.5, 0.5]), probabilities, 1e-6)
  }

  @Test
  fun testPredictProbabilityDataFrame() {
    val csvData = """
      a,b,c
      1,2,0
      2,4,1
      3,8,1
      """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = LogisticRegression().fit(X, y)
    val probabilities = model.predictProbability(X)

    assertEquals(listOf(0.5, 0.5, 0.5), probabilities.toList(), 1e-6)
  }
}
