package com.github.aaarrti.tl2jgen

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.nio.file.Files
import java.nio.file.Paths

object ResourceReader {


    /**
     * @return list of (input features, expected output)
     */
    fun readResourceFile(fileName: String): List<Pair<List<Double>, Double>> {

        val resource = javaClass.classLoader.getResource(fileName)
            ?: throw IllegalArgumentException("Resource not found: $fileName")

        val path = Paths.get(resource.toURI())
        val content = Files.readString(path)

        val data = Json.decodeFromString<DataSample>(content)
        return data.x.zip(data.y)
    }


    @Serializable
    private data class DataSample(
        @SerialName("X") val x: List<List<Double>>,
        @SerialName("y_pred") val y: List<Double>,
    )


}

