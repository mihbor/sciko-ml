package ltd.mbor.sciko.ml

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

fun DataFrame.trainTestSplit(testSize: Double): Pair<DataFrame, DataFrame> {
  val testSizeInt = (testSize * rowCount).toInt()
  val shuffledIndices = (0 until rowCount).shuffled()
  val testIndices = shuffledIndices.take(testSizeInt)
  val trainIndices = shuffledIndices.drop(testSizeInt)
  return Pair(
    DataFrame(columns.map { column ->
      DataFrame.Column(column.name, column.type, column.values.filterIndexed { index, _ -> index in trainIndices })
    }),
    DataFrame(columns.map { column ->
      DataFrame.Column(column.name, column.type, column.values.filterIndexed { index, _ -> index in testIndices })
    })
  )
}

fun MultiArray<Double, D2>.trainTestSplit(testSize: Double): Pair<MultiArray<Double, D2>, MultiArray<Double, D2>> {
  val testSizeInt = (testSize * shape[0]).toInt()
  val shuffledIndices = (0 until shape[0]).shuffled()
  val testIndices = shuffledIndices.take(testSizeInt)
  val trainIndices = shuffledIndices.drop(testSizeInt)
  return Pair(
    mk.ndarray(trainIndices.map { get(it).toList() }),
    mk.ndarray(testIndices.map { get(it).toList() })
  )
}