package ltd.mbor.sciko.ml.tree

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.test.Test
import kotlin.test.assertEquals

class DecisionTreeRegressorTest {

    @Test
    fun testFit() {
        val X = mk.ndarray(mk[
            mk[1.0, 2.0],
            mk[2.0, 4.0],
            mk[3.0, 8.0]
        ])
        val y = mk.ndarray(mk[4.0, 7.0, 12.0])

        val model = DecisionTreeRegressor().fit(X, y)

        assertEquals(4.0, model.predict(X[0]), 1e-6)
        assertEquals(7.0, model.predict(X[1]), 1e-6)
        assertEquals(12.0, model.predict(X[2]), 1e-6)
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

        val model = DecisionTreeRegressor().fit(X, y)

        assertEquals(4.0, model.predict(X[0]), 1e-6)
        assertEquals(7.0, model.predict(X[1]), 1e-6)
        assertEquals(12.0, model.predict(X[2]), 1e-6)
    }

    @Test
    fun testPredict() {
        val X = mk.ndarray(mk[
            mk[1.0, 2.0],
            mk[2.0, 4.0],
            mk[3.0, 8.0]
        ])
        val y = mk.ndarray(mk[4.0, 7.0, 12.0])

        val model = DecisionTreeRegressor().fit(X, y)
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

        val model = DecisionTreeRegressor().fit(X, y)
        val predictions = model.predict(X)

        assertEquals(y.values, predictions.toList(), 1e-6)
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
