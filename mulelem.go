// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
)

// MulElem represents element-wise multiplication.
type MulElem struct {
	Left  MatrixExp
	Right MatrixExp
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

// Eval returns a matrix literal.
func (m1 *MulElem) Eval() MatrixLiteral {
	r, c := m1.Dims()

	lm := m1.Left.Eval()
	rm := m1.Right.Eval()

	v1 := lm.AsVector()
	v2 := rm.AsVector()
	for i, v := range v2 {
		v1[i] *= v
	}

	return &General{blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   v1,
	}}
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *MulElem) Copy() MatrixExp {
	return &MulElem{
		Left:  m1.Left.Copy(),
		Right: m1.Right.Copy(),
	}
}

// T transposes a matrix.
func (m1 *MulElem) T() MatrixExp {
	return &T{m1}
}

// Add two matrices together.
func (m1 *MulElem) Add(m2 MatrixExp) MatrixExp {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *MulElem) Sub(m2 MatrixExp) MatrixExp {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *MulElem) Scale(c float64) MatrixExp {
	return &Scale{
		C: c,
		M: m1,
	}
}

// Mul performs matrix multiplication.
func (m1 *MulElem) Mul(m2 MatrixExp) MatrixExp {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *MulElem) MulElem(m2 MatrixExp) MatrixExp {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *MulElem) DivElem(m2 MatrixExp) MatrixExp {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
