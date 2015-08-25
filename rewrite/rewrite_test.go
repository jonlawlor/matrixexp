// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package rewrite

import (
	"github.com/gonum/blas/blas64"
	"github.com/jonlawlor/matrixexp"
	"math/rand"
	"testing"
)

// Functions to help generate example matrix literals.
func GeneralZeros(r, c int) matrixexp.MatrixExp {
	return &matrixexp.General{zeros(r, c)}
}
func zeros(r, c int) blas64.General {
	return blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   make([]float64, r*c),
	}
}

func GeneralOnes(r, c int) matrixexp.MatrixExp {
	return &matrixexp.General{ones(r, c)}
}
func ones(r, c int) blas64.General {
	m := zeros(r, c)
	for i := range m.Data {
		m.Data[i] = 1
	}
	return m
}

func GeneralRand(r, c int) matrixexp.MatrixExp {
	return &matrixexp.General{rnd(r, c)}
}
func rnd(r, c int) blas64.General {
	m := zeros(r, c)
	rs := rand.New(rand.NewSource(99))
	for i := range m.Data {
		m.Data[i] = rs.NormFloat64()
	}
	return m
}

func eye(r int) blas64.General {
	g := zeros(r, r)
	for i := 0; i < r*r; i += r + 1 {
		g.Data[i] = 1
	}
	return g
}

func TestRewrite(t *testing.T) {
	// Generate an example rewrite rule: (AnyA.T()).Add(AnyB.T()) -> (AnyA.Add(AnyB)).T()
	AnyA := &AnyExp{}
	AnyB := &AnyExp{}
	AddRewrite := Template((AnyA.T()).Add(AnyB.T()), (AnyA.Add(AnyB).T()))

	ExA := GeneralZeros(10, 1)
	ExB := GeneralOnes(10, 1)
	ExFrom := (ExA.T()).Add(ExB.T())

	ExTo, err := AddRewrite.Rewrite(ExFrom)

	if err != nil {
		t.Errorf("non-nil error encountered during rewrite: %v", err)
	}
	// Check that ExFrom and ExTo are equivalent
	if v := matrixexp.Equals(ExFrom, ExTo); v != true {
		t.Errorf("Equals(%v,%v) equals %v, want %v", ExFrom, ExTo, v, true)
	}

}
