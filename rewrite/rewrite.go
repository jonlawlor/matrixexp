// Copyright 2015 Jonathan J Lawlor. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package rewrite

import (
	"fmt"
	"github.com/jonlawlor/matrixexp"
	"reflect"
)

// A Compiler can transform matrix expressions by applying various optimizations
// to it, such as converting a.T().Mul(b.T()) to an appropriate call to DGEMM,
// or allowing concurrent evaluation of (a.Mul(b)).Add(c.Mul(d)) via Async.
type Compiler interface {
	// Compile transforms a matrix, or returns an error indicating a problem
	// with either the matrix expression or the compiler.
	Compile(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)

	// MustCompile is just like Compile, except that instead of returning an error
	// it will panic if it encounters a problem.  It is intended for use in init()
	MustCompile(matrixexp.MatrixExp) matrixexp.MatrixExp
}

// Rewriter can convert one matrix expression to another.
type Rewriter interface {
	Rewrite(matrixexp.MatrixExp) (matrixexp.MatrixExp, error)
}

// A Matcher can determine if an expression wildcard matches another matrix
// expression.  If the expression does not match the wildcard then it returns
// an error explaining why.
type Matcher interface {
	Match(matrixexp.MatrixExp) error
}

// template represents an example based rule for the compiler to follow.
type template struct {
	from matrixexp.MatrixExp
	to   matrixexp.MatrixExp
}

// Template produces a rewrite rule from a matrix expression template.
func Template(from, to matrixexp.MatrixExp) Rewriter {
	return &template{
		from: from,
		to:   to,
	}
}

// Rewrite applies a rewrite rule to a matrix expression.  If it can't be applied
// to the input expression, then it returns an error.
func (r *template) Rewrite(m1 matrixexp.MatrixExp) (matrixexp.MatrixExp, error) {

	// Determine if the matrix expression matches the rewrite rule.
	// TODO(jonlawlor): implement some kind of reflection cache.
	m2 := m1.Copy()
	matMap := make(map[matrixexp.MatrixExp]matrixexp.MatrixExp)
	if err := matches(m1, r.from, matMap); err != nil {
		// TODO(jonlawlor): implement useful error message.  This is just a placeholder.
		// We'll probably need a better way of converting a matrix expression to a
		// string.
		return nil, err
	}
	return m2, nil
}

// matches returns true if the matrix expression matches the template.  It also
// modifies the map to contain the mapping between the input matrix expression
// and the template output.
func matches(m1, from matrixexp.MatrixExp, matMap map[matrixexp.MatrixExp]matrixexp.MatrixExp) error {
	if w, ok := from.(Matcher); ok {
		// from is a wildcard, so it can be directly compared
		if err := w.Match(m1); err != nil {
			return err
		}
		// Determine if we have seen the expression before.
		if to, seen := matMap[from]; !seen {
			matMap[from] = m1
		} else if seen && m1 != to {
			return &NewExpMismatch{m1, to}
		}
		// I'm not sure if a wlldcard should have subexpressions.  For now assume
		// no, and we can always add the capability later.
		return nil
	}
	// Determine if m1 and from are the same matrix operation.
	rm1 := reflect.ValueOf(m1)
	rfrom := reflect.ValueOf(from)
	if rm1.Type() != rfrom.Type() {
		return &ExpMismatch{from, m1}
	}
	// Walk subexpressions.
	rm1 = follow(rm1)
	rfrom = follow(rfrom)
	for i := 0; i < rfrom.NumField(); i++ {
		// if rfrom is a matrix expression, call matches on it as well
		if f := rfrom.Field(i); f.Type().Implements(reflect.TypeOf((*matrixexp.MatrixExp)(nil)).Elem()) {
			if err := matches(f.Interface().(matrixexp.MatrixExp), rm1.Field(i).Interface().(matrixexp.MatrixExp), matMap); err != nil {
				return err
			}
		}
	}
	return nil
}

func follow(r1 reflect.Value) reflect.Value {
	for ; r1.Kind() == reflect.Ptr; r1 = r1.Elem() {
	}
	return r1
}

// ExpMismatch indicates that an expression does not match an expected pattern.
type ExpMismatch struct {
	expected matrixexp.MatrixExp
	got      matrixexp.MatrixExp
}

// Error implements the error interface.
func (e *ExpMismatch) Error() string {
	et := reflect.TypeOf(e.expected)
	gt := reflect.TypeOf(e.got)
	return "expression type mismatch: expected " + et.PkgPath() + "/" + et.Name() + " got: " + gt.PkgPath() + "/" + gt.Name()
}

// NewExpMismatch indicates that a wildcard was expected to be repeated in the
// expression tree, but a new matrix expressio was found.  For example, the
// expression A.Mul(A.T()) would return this when compared with B.Mul(C.T())
type NewExpMismatch struct {
	expected matrixexp.MatrixExp
	got      matrixexp.MatrixExp
}

// Error implements the error interface.
func (e *NewExpMismatch) Error() string {
	return fmt.Sprintf("expected previously seen expression %v, got new %v", e.expected, e.got)
}
