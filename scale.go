// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
	"strconv"
)

// Scale represents scalar multiplication.
type Scale struct {
	C float64
	M MatrixExp
}

// String implements the Stringer interface.
func (m1 *Scale) String() string {
	return m1.M.String() + ".Scale(" + strconv.FormatFloat(m1.C, 'g', -1, 64) + ")"
}

// Dims returns the matrix dimensions.
func (m1 *Scale) Dims() (r, c int) {
	r, c = m1.M.Dims()
	return
}

// At returns the value at a given row, column index.
func (m1 *Scale) At(r, c int) float64 {
	return m1.M.At(r, c) * m1.C
}

// Eval returns a matrix literal.
func (m1 *Scale) Eval() MatrixLiteral {
	r, c := m1.Dims()

	mv := m1.M.Eval()
	v1 := mv.AsVector()
	C := m1.C
	for i := range v1 {
		v1[i] *= C
	}

	return &General{blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   v1,
	}}
}

// Copy creates a (deep) copy of the Matrix Expression.
func (m1 *Scale) Copy() MatrixExp {
	return &Scale{
		C: m1.C,
		M: m1.M,
	}
}

// Err returns the first error encountered while constructing the matrix expression.
func (m1 *Scale) Err() error {
	return m1.M.Err()
}

// T transposes a matrix.
func (m1 *Scale) T() MatrixExp {
	return &T{m1}
}

// Add two matrices together.
func (m1 *Scale) Add(m2 MatrixExp) MatrixExp {
	return &Add{
		Left:  m1,
		Right: m2,
	}
}

// Sub subtracts the right matrix from the left matrix.
func (m1 *Scale) Sub(m2 MatrixExp) MatrixExp {
	return &Sub{
		Left:  m1,
		Right: m2,
	}
}

// Scale performs scalar multiplication.
func (m1 *Scale) Scale(c float64) MatrixExp {
	return &Scale{
		C: c * m1.C,
		M: m1.M,
	}
}

// Mul performs matrix multiplication.
func (m1 *Scale) Mul(m2 MatrixExp) MatrixExp {
	return &Mul{
		Left:  m1,
		Right: m2,
	}
}

// MulElem performs element-wise multiplication.
func (m1 *Scale) MulElem(m2 MatrixExp) MatrixExp {
	return &MulElem{
		Left:  m1,
		Right: m2,
	}
}

// DivElem performs element-wise division.
func (m1 *Scale) DivElem(m2 MatrixExp) MatrixExp {
	return &DivElem{
		Left:  m1,
		Right: m2,
	}
}
