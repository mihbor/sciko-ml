package ltd.mbor.sciko.ml

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.test.Test
import kotlin.test.assertEquals

class LinearRegressionTest {

    @Test
    fun testFit() {
        val X = mk.ndarray(mk[
            mk[1.0, 2.0],
            mk[2.0, 3.0],
            mk[3.0, 4.0]
        ])
        val y = mk.ndarray(mk[3.0, 5.0, 7.0])

        val model = LinearRegression().fit(X, y)

        assertEquals(1.0, model.bias, 1e-6)
        assertEquals(mk.ndarray(mk[1.0, 1.0]), model.weights)
    }

    @Test
    fun testPredict() {
        val X = mk.ndarray(mk[
            mk[1.0, 2.0],
            mk[2.0, 3.0],
            mk[3.0, 4.0]
        ])
        val y = mk.ndarray(mk[3.0, 5.0, 7.0])

        val model = LinearRegression().fit(X, y)
        val predictions = model.predict(X)

        assertEquals(y, predictions)
    }
}
