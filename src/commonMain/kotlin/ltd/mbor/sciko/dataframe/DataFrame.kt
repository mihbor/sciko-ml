package ltd.mbor.sciko.dataframe

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader

class DataFrame(val columns: List<Column<*>>) {
  enum class DataType {
    DOUBLE,
    STRING,
    BOOLEAN
  }
  class Column<T>(
    val name: String,
    val type: DataType,
    val values: List<T>
  )

  companion object{
    fun readCsv(csvData: String): DataFrame {
      val rows: List<List<String>> = csvReader().readAll(csvData)
      val columnCount = rows.first().size
      val columns = List(columnCount) { i ->
        val name = rows.first()[i]
        val values = rows.drop(1).map { row -> row[i] }
        when {
          values.all { it.toDoubleOrNull() != null } -> Column(name, DataType.DOUBLE, values.map { it.toDouble() })
          values.all { it.toBooleanStrictOrNull() != null } -> Column(name, DataType.BOOLEAN, values.map { it.toBooleanStrict() })
          else -> Column(name, DataType.STRING, values)
        }
      }
      return DataFrame(columns)
    }
  }
}