package ltd.mbor.sciko.ml

import ltd.mbor.sciko.ml.dataframe.DataFrame

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