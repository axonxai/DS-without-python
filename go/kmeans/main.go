package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"github.com/muesli/kmeans/plotter"
)

func main() {
	// Open the dataset file.
	f, err := os.Open("../../data/clustering/clustering.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader.
	r := csv.NewReader(f)
	r.FieldsPerRecord = 2

	// set up a random two-dimensional data set (float64 values between 0.0 and 1.0)
	var d clusters.Observations

	// Loop over the records creating our slice of
	// gokmeans.Nodes.
	for {

		// Read in our record and check for errors.
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		// Initialize a point.
		var point []float64

		// Fill in our point.
		for i := 0; i < 2; i++ {
			val, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				log.Fatal(err)
			}

			// Append this value to our point
			point = append(point, val)
		}

		// Append our point to the data.
		d = append(d, clusters.Coordinates{
			point[0],
			point[1],
		})
	}

	// Partition the data points into 4 clusters
	km, err := kmeans.NewWithOptions(0.01, plotter.SimplePlotter{})
	if err != nil {
		log.Fatal(err)
	}

	clusters, err := km.Partition(d, 4)
	if err != nil {
		log.Fatal(err)
	}

	for _, c := range clusters {
		fmt.Printf("Centered at x: %.2f y: %.2f\n", c.Center[0], c.Center[1])
		fmt.Printf("Matching data points: %+v\n\n", c.Observations)
	}
}
