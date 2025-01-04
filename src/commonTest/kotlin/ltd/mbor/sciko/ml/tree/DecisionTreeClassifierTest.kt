package ltd.mbor.sciko.ml.tree

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class DecisionTreeClassifierTest {

  @Test
  fun testFit() {
    val X = mk.ndarray(
      mk[
        mk[1.0, 2.0],
        mk[2.0, 4.0],
        mk[3.0, 8.0]
      ]
    )
    val y = mk.ndarray(mk[0, 1, 0])

    val model = DecisionTreeClassifier().fit(X, y)

    assertEquals(0, model.predict(X[0]))
    assertEquals(1, model.predict(X[1]))
    assertEquals(0, model.predict(X[2]))
  }

  @Test
  fun testFitDataFrame() {
    val csvData = """
      a,b,c
      1,2,0
      2,4,1
      3,8,0
      """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = DecisionTreeClassifier().fit(X, y)

    assertEquals(0, model.predict(X[0] as List<Double>))
    assertEquals(1, model.predict(X[1] as List<Double>))
    assertEquals(0, model.predict(X[2] as List<Double>))
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
    val y = mk.ndarray(mk[0, 1, 0])

    val model = DecisionTreeClassifier().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y, predictions)
  }

  @Test
  fun testPredictDataFrame() {
    val csvData = """
      a,b,c
      1,2,0
      2,4,1
      3,8,0
      """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val X = dataFrame.dropColumnsByName("c")
    val y = dataFrame["c"] as DataFrame.Column<Double>

    val model = DecisionTreeClassifier().fit(X, y)
    val predictions = model.predict(X)

    assertEquals(y.values.map { it.toInt() }, predictions.toList())
  }
}
