// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

// T represents a transposed matrix expression.
type T struct {
	Matrix
}

// Dims returns the matrix dimensions.
func (m *T) Dims() (r, c int) {
	c, r = m.Matrix.Dims()
	return
}

// At returns the value at a given row, column index.
func (m *T) At(r, c int) float64 {
	return m.Matrix.At(c, r)
}

// Set changes the value at a given row, column index.
func (m *T) Set(r, c int, v float64) {
	m.Matrix.Set(c, r, v)
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m *T) Vector() []float64 {
	mr, mc := m.Matrix.Dims()
	mv := m.Matrix.Vector()

	// This is the naive implementation.
	// TODO(jonlawlor): implement something smarter, such as the cache oblivious
	// algorithm explained on wikipedia.
	v := make([]float64, len(mv))
	for i := 0; i < mr; i++ {
		for j := 0; j < mc; j++ {
			v[j*mr+i] = mv[i*mc+j]
		}
	}
	return v
}
