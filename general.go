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
func (m *General) Dims() (r, c int) {
	r, c = m.Rows, m.Cols
	return
}

// At returns the value at a given row, column index.
func (m *General) At(r, c int) float64 {
	return m.Data[r*m.Stride+c]
}

// Set changes the value at a given row, column index.
func (m *General) Set(r, c int, v float64) {
	m.Data[r*m.Stride+c] = v
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m *General) Vector() []float64 {
	v := make([]float64, len(m.Data))
	copy(v, m.Data)
	return v
}
