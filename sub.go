// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// Sub represents matrix subtraction.
type Sub struct {
	Left  Matrix
	Right Matrix
}

// Dims returns the matrix dimensions.
func (m1 *Sub) Dims() (r, c int) {
	r, c = m1.Left.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *Sub) At(r, c int) float64 {
	return m1.Left.At(r, c) - m1.Right.At(r, c)
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *Sub) Vector() []float64 {
	v1 := m1.Left.Vector()
	v2 := m1.Right.Vector()
	for i, v := range v2 {
		v1[i] -= v
	}
	return v1
}

// Eval returns a matrix literal.
func (m1 *Sub) Eval() Matrix {
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
func (m1 *Sub) T() Matrix {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Sub) Add(m2 Matrix) Matrix {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Sub) Sub(m2 Matrix) Matrix {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Mul performs matrix multiplication.
func (m1 *Sub) Mul(m2 Matrix) Matrix {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Sub) MulElem(m2 Matrix) Matrix {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Sub) DivElem(m2 Matrix) Matrix {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
