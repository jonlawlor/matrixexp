// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// Sub represents matrix subtraction.
type Sub struct {
	Left  MatrixExpr
	Right MatrixExpr
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

// Eval returns a matrix literal.
func (m1 *Sub) Eval() MatrixLiteral {
	r, c := m1.Dims()

	lm := m1.Left.Eval()
	rm := m1.Right.Eval()

	v1 := lm.AsVector()
	v2 := rm.AsVector()
	for i, v := range v2 {
		v1[i] -= v
	}

	return &General{blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   v1,
	}}
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *Sub) Copy() MatrixExpr {
	return &Sub{
		Left:  m1.Left.Copy(),
		Right: m1.Right.Copy(),
	}
}

// T transposes a matrix.
func (m1 *Sub) T() MatrixExpr {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Sub) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Sub) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *Sub) Scale(c float64) MatrixExpr {
	return &Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *Sub) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Sub) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Sub) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
