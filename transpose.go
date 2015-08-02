// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// T represents a transposed matrix expression.
type T struct {
	M MatrixExpr
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

// Eval returns a matrix literal.
func (m1 *T) Eval() MatrixLiteral {
	mr, mc := m1.M.Dims()
	mv := m1.M.Eval().AsVector()
	v := make([]float64, len(mv))
	for i := 0; i < mr; i++ {
		for j := 0; j < mc; j++ {
			v[j*mr+i] = mv[i*mc+j]
		}
	}

	return &General{blas64.General{
		Rows:   mc,
		Cols:   mr,
		Stride: mr,
		Data:   v,
	}}
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *T) Copy() MatrixExpr {
	return &T{
		M: m1.M.Copy(),
	}
}

// T transposes a matrix.
func (m1 *T) T() MatrixExpr {
	return m1.M
}

// Add two matrices together.
func (m1 *T) Add(m2 MatrixExpr) MatrixExpr {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *T) Sub(m2 MatrixExpr) MatrixExpr {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *T) Scale(c float64) MatrixExpr {
	return &Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *T) Mul(m2 MatrixExpr) MatrixExpr {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *T) MulElem(m2 MatrixExpr) MatrixExpr {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *T) DivElem(m2 MatrixExpr) MatrixExpr {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
