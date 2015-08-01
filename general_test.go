// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package matrixexp

import (
	"github.com/gonum/blas/blas64"
	"math/rand"
	"testing"
)

type MatrixGenerator struct {
	r, c int
	gen  func(r, c int) MatrixExpr
	want blas64.General
}

// Functions to help generate example matrices.
func GeneralZeros(r, c int) MatrixExpr {
	return &General{zeros(r, c)}
}
func zeros(r, c int) blas64.General {
	return blas64.General{
		Rows:   r,
		Cols:   c,
		Stride: c,
		Data:   make([]float64, r*c),
	}
}

func GeneralOnes(r, c int) MatrixExpr {
	return &General{ones(r, c)}
}
func ones(r, c int) blas64.General {
	m := zeros(r, c)
	for i := range m.Data {
		m.Data[i] = 1
	}
	return m
}

func GeneralRand(r, c int) MatrixExpr {
	return &General{rnd(r, c)}
}
func rnd(r, c int) blas64.General {
	m := zeros(r, c)
	rs := rand.New(rand.NewSource(99))
	for i := range m.Data {
		m.Data[i] = rs.NormFloat64()
	}
	return m
}

var GeneralMatrices = []MatrixGenerator{
	{r: 1, c: 1, gen: GeneralZeros, want: zeros(1, 1)},
	{r: 10, c: 1, gen: GeneralZeros, want: zeros(10, 1)},
	{r: 1, c: 10, gen: GeneralZeros, want: zeros(1, 10)},
	{r: 10, c: 10, gen: GeneralZeros, want: zeros(10, 10)},

	{r: 1, c: 1, gen: GeneralOnes, want: ones(1, 1)},
	{r: 10, c: 1, gen: GeneralOnes, want: ones(10, 1)},
	{r: 1, c: 10, gen: GeneralOnes, want: ones(1, 10)},
	{r: 10, c: 10, gen: GeneralOnes, want: ones(10, 10)},

	{r: 1, c: 1, gen: GeneralRand, want: rnd(1, 1)},
	{r: 10, c: 1, gen: GeneralRand, want: rnd(10, 1)},
	{r: 1, c: 10, gen: GeneralRand, want: rnd(1, 10)},
	{r: 10, c: 10, gen: GeneralRand, want: rnd(10, 10)},
}

func TestGeneral(t *testing.T) {
	for i, tt := range GeneralMatrices {
		m := tt.gen(tt.r, tt.c)
		r, c := m.Dims()
		if r != tt.r {
			t.Errorf("%d: %q rows are %d, want %d", i, m, r, tt.r)
		}
		if c != tt.c {
			t.Errorf("%d: %q columns are %d, want %d", i, m, c, tt.c)
		}
		want := &General{tt.want}
		if got := m.Eval(); !Equals(got, want) {
			t.Errorf("%d: matrix %q equals %q, want %q", i, m, got, want)
		}
	}
}
