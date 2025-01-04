package ltd.mbor.sciko.ml.dataframe

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader

class DataFrame(val columns: List<Column<*>>, val rowCount: Int = columns.first().values.size) {
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
          values.all { it.toDoubleOrNull() != null || it.isEmpty() } -> Column(name, DataType.DOUBLE, values.map { it.toDoubleOrNull() })
          values.all { it.toBooleanStrictOrNull() != null || it.isEmpty() } -> Column(name, DataType.BOOLEAN, values.map { it.toBooleanStrictOrNull() })
          else -> Column(name, DataType.STRING, values)
        }
      }
      return DataFrame(columns)
    }
  }

  operator fun get(columnName: String): Column<*>? {
    return columns.singleOrNull { it.name == columnName }
  }

  operator fun get(index: Int): List<Any?> {
    return columns.map { it.values[index] }
  }

  fun dropNA(columnNames: List<String>): DataFrame {
    val columnsToExamine = columns.filter{it.name in columnNames}
    val rowsToDrop = mutableSetOf<Int>()
    columnsToExamine.forEach { column ->
      column.values.forEachIndexed { index, value ->
        if (value == null) {
          rowsToDrop.add(index)
        }
      }
    }
    return DataFrame(columns.map { column ->
      Column(column.name, column.type, column.values.filterIndexed { index, _ -> index !in rowsToDrop })
    })
  }

  fun dropColumnsByName(vararg columnNames: String): DataFrame {
    return DataFrame(columns.filter { it.name !in columnNames })
  }

  fun dropColumnsByIndex(columnIndices: Collection<Int>): DataFrame {
    return DataFrame(columns.filterIndexed { index, _ -> index !in columnIndices })
  }

  fun columnIndicesOfTypes(vararg types: DataType): List<Int> {
    return columns.mapIndexedNotNull { index, column ->
      if (column.type in types) index else null
    }
  }

  fun columnIndicesNotOfTypes(vararg types: DataType): List<Int> {
    return columns.mapIndexedNotNull { index, column ->
      if (column.type !in types) index else null
    }
  }
}