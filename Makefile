# SYNOPSIS:
#
#   make [all]  - makes everything.
#   make test   - makes everything and runs unit tests.
#   make TARGET - makes the given target.
#   make clean  - removes all files generated by make.

# Please tweak the following variable definitions as needed by your
# project, except GTEST_HEADERS, which you can use in your own targets
# but shouldn't modify.

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = /usr/src/googletest/googletest

# Where to find test source
USER_TEST_DIR = $(dir $(lastword $(MAKEFILE_LIST)))/test

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include -Iinclude

# Flags passed to the C++ compiler.
CXXFLAGS += -g2 -Wall -Wextra -pthread

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS = bin/test/cache_unittest

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# House-keeping build targets.

all : $(TESTS)

test : all
	bin/test/cache_unittest

clean :
	rm -f $(TESTS) bin/test/gtest.a bin/test/gtest_main.a bin/test/*.o bin/*.o

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
bin/test/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc -o $@

bin/test/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc -o $@

bin/test/gtest.a : bin/test/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

bin/test/gtest_main.a : bin/test/gtest-all.o bin/test/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

bin/test/cache_unittest.o : $(USER_TEST_DIR)/cache_unittest.cc include/*.h include/sketches/*.h $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_TEST_DIR)/cache_unittest.cc -o $@

bin/test/cache_unittest : bin/test/cache_unittest.o bin/test/gtest.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread $^ -o $@