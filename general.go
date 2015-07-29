// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// General is a typical matrix literal
type General struct {
	blas64.General
}

// Dims returns the matrix dimensions.
func (g *General) Dims() (r, c int) {
	r, c = g.Rows, g.Cols
	return
}

// At returns the value at a given row, column index.
func (g *General) At(r, c int) float64 {
	return g.Data[r*g.Stride+c]
}

// Set changes the value at a given row, column index.
func (g *General) Set(r, c int, v float64) {
	g.Data[r*g.Stride+c] = v
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (g *General) Vector() []float64 {
	v := make([]float64, len(g.Data))
	copy(v, g.Data)
	return v
}
