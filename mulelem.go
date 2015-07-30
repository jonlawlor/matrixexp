// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

type MulElem struct {
	Left  Matrix
	Right Matrix
}

// Dims returns the matrix dimensions.
func (m1 *MulElem) Dims() (r, c int) {
	r, c = m1.Left.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *MulElem) At(r, c int) float64 {
	return m1.Left.At(r, c) * m1.Right.At(r, c)
}

// Vector returns all of the values in the matrix as a []float64, in row order.
func (m1 *MulElem) Vector() []float64 {
	v1 := m1.Left.Vector()
	v2 := m1.Right.Vector()
	for i, v := range v2 {
		v1[i] *= v
	}
	return v1
}

func (m1 *MulElem) Eval() Matrix {
	r, c := m1.Dims()
	v := m1.Vector()
	return &General{blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   v,
	}}
}

func (m1 *MulElem) T() Matrix {
	return &T{m1}
}

func (m1 *MulElem) Add(m2 Matrix) Matrix {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *MulElem) Sub(m2 Matrix) Matrix {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *MulElem) Mul(m2 Matrix) Matrix {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *MulElem) MulElem(m2 Matrix) Matrix {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

func (m1 *MulElem) DivElem(m2 Matrix) Matrix {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
