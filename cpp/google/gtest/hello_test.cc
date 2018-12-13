#include <iostream>
#include <gtest/gtest.h>
#include "hello.h"

TEST(BasicTest, HandleNoneZeroInput) {
	EXPECT_EQ(2, Foo(4,10));
	EXPECT_EQ(6, Foo(30,18));
}

bool MutuallyPrime(int m, int n) {
	return Foo(m, n) > 1;
}

template<typename T>
class FooType {
public:
	void Bar() {
		testing::StaticAssertTypeEq<int,T>();
	}
};

// global event
class FooEnvironment: public testing::Environment {
public:
	virtual void SetUp() {
		std::cout << "Foo FooEnvironment SetUp" << std::endl;
	}

	virtual void TearDown() {
		std::cout << "Foo FooEnvironment TearDown" << std::endl;
	}
};

// TestSuite event
class FooTest:public testing::Test {
protected:
	static void SetUpTestCase() {
		counter_++;
		std::cout << "Foo FooTest SetUpTestCase[" << counter_ << "]" << std::endl;
	}

	static void TearDownTestCase() {
		std::cout << "Foo FooTest TearDownTestCase[" << counter_ << "]" << std::endl;
	}

	static int counter_;
};

int FooTest::counter_ = 0;

TEST_F(FooTest, Test1) {

}

TEST_F(FooTest, Test2) {

}


// TestCase event
class FooCalcTest:public testing::Test {
protected:
	virtual void SetUp() {
		std::cout << "Foo FooCalcTest Setup" << std::endl;
	}

	virtual void TearDown() {
		std::cout << "Foo FooCalcTest TearDown" << std::endl;
	}
};

TEST_F(FooCalcTest, HandleNonZeroInput) {
	EXPECT_EQ(4, Foo(12,8));
}

TEST_F(FooCalcTest, HandleExpcetion) {
	EXPECT_ANY_THROW(Foo(0,100));
	EXPECT_ANY_THROW(Foo(0,0));
	EXPECT_ANY_THROW(Foo(100,0));
}


TEST(BasicTest, BasicGTest) {
	// boolean
	ASSERT_TRUE(1 < 2);
	ASSERT_FALSE(1 > 2);

	// number
	ASSERT_EQ(1+2,4-1); // parameter:(expected,actual)
	ASSERT_NE(1+2,1-2);
	ASSERT_LT(1,2);
	ASSERT_LE(1,1);
	ASSERT_GT(2,1);
	ASSERT_GE(2,2);

	// string
	ASSERT_STREQ("hello","hello");
	ASSERT_STRNE("hello","hellO");
	ASSERT_STRCASEEQ("hello","HeLLo");
	ASSERT_STRCASENE("hello","hallo");

	// explicit success or failure
	// ADD_FAILURE() << "Sorry";

	// exception
	EXPECT_ANY_THROW(Foo(10,0));
	EXPECT_THROW(Foo(0,5),const char*);

	// predicate assertion
	int m = 12,n = 6;
	EXPECT_PRED2(MutuallyPrime, m, n); // MutuallyPrime(m,n) -> true

	// float
	ASSERT_FLOAT_EQ(3.14f,3.14f);
	ASSERT_DOUBLE_EQ(3.14,3.14);

	EXPECT_NEAR(3.14f,3.13f,0.3f);
	
	// type check
	FooType<int> fooType;
	fooType.Bar();
}


class IsPrimeParamTest:public testing::TestWithParam<int> {

};

TEST_P(IsPrimeParamTest, HandleTrueReturn) {
	int n = GetParam();
	EXPECT_TRUE(IsPrime(n));
}
INSTANTIATE_TEST_CASE_P(TrueReturn, IsPrimeParamTest, testing::Values(3,5,11,23,17));

int main(int argc, char *argv[]) {

	testing::AddGlobalTestEnvironment(new FooEnvironment);
	testing::InitGoogleTest(&argc,argv);
	return RUN_ALL_TESTS();
}

