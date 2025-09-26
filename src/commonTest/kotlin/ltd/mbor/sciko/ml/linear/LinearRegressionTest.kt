package ltd.mbor.sciko.ml.linear

import ltd.mbor.sciko.ml.assertEquals
import ltd.mbor.sciko.ml.dataframe.DataFrame
import ltd.mbor.sciko.ml.metrics.r2Score
import ltd.mbor.sciko.ml.trainTestSplit
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

  @Test
  fun testMtCars() {
    val dataFrame = DataFrame.readCsv(this::class.java.getResource("/mtcars.csv").readText())
    val (train, test) = dataFrame.trainTestSplit(0.18, Random(14))
    val X = train.keepColumnsByName("hp", "wt")
    val y = train.get("mpg") as DataFrame.Column<Double>
    val model = LinearRegression().fit(X, y)
    val pred = model.predict(test.keepColumnsByName("hp", "wt"))
    val r2 = r2Score(test.get("mpg") as DataFrame.Column<Double>, pred)
    println("R2: $r2")
    assertTrue(r2 > 0.72, "Fit should have an R2 score of just over 0.72")
    assertTrue(r2 < 0.720015, "Fit should have an R2 score of just over 0.72")
  }
}
