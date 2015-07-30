// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// T represents a transposed matrix expression.
type T struct {
	M Matrix
}

// Dims returns the matrix dimensions.
func (m1 *T) Dims() (r, c int) {
	c, r = m1.M.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *T) At(r, c int) float64 {
	return m1.M.At(c, r)
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *T) Vector() []float64 {
	mr, mc := m1.M.Dims()
	mv := m1.M.Vector()

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

// Eval returns a matrix literal.
func (m1 *T) Eval() Matrix {
	r, c := m1.Dims()
	v := m1.Vector()
	return &General{blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   v,
	}}
}

// T transposes a matrix.
func (m1 *T) T() Matrix {
	return m1.M
}

// Add two matrices together.
func (m1 *T) Add(m2 Matrix) Matrix {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *T) Sub(m2 Matrix) Matrix {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Mul performs matrix multiplication.
func (m1 *T) Mul(m2 Matrix) Matrix {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *T) MulElem(m2 Matrix) Matrix {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *T) DivElem(m2 Matrix) Matrix {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
