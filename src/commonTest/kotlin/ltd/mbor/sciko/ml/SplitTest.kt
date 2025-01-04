package ltd.mbor.sciko.ml

import ltd.mbor.sciko.ml.dataframe.DataFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import kotlin.test.Test
import kotlin.test.assertEquals

class SplitTest {

  @Test
  fun testTrainTestSplit() {
    val X = mk.ndarray(mk[
      mk[1.0, 2.0],
      mk[2.0, 4.0],
      mk[3.0, 8.0],
      mk[4.0, 16.0]
    ])

    val (train, test) = X.trainTestSplit(0.5)

    assertEquals(2, test.shape[0])
    assertEquals(2, train.shape[0])

    // Check that all columns are present in both splits
    assertEquals(X.shape[1], train.shape[1])
    assertEquals(X.shape[1], test.shape[1])

    // Check that the sum of rows in train and test sets equals the original row count
    assertEquals(X.shape[0], train.shape[0] + test.shape[0])
  }

  @Test
  fun testTrainTestSplitDataFrame() {
    val csvData = """
            name,age,active
            John,30,true
            Jane,25,false
            Bob,40,true
            Alice,35,false
        """.trimIndent()

    val dataFrame = DataFrame.readCsv(csvData)
    val (trainDataFrame, testDataFrame) = dataFrame.trainTestSplit(0.5)

    assertEquals(2, testDataFrame.rowCount)
    assertEquals(2, trainDataFrame.rowCount)

    // Check that all columns are present in both splits
    assertEquals(dataFrame.columns.size, trainDataFrame.columns.size)
    assertEquals(dataFrame.columns.size, testDataFrame.columns.size)

    // Check that the sum of rows in train and test sets equals the original row count
    assertEquals(dataFrame.rowCount, trainDataFrame.rowCount + testDataFrame.rowCount)
  }
}