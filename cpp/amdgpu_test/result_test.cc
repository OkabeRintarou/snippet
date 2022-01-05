#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "result.h"


Result<int, float> func0(int arg) {
    if (arg == 0) {
        return Ok(100);
    } else {
        return Err(3.14f);
    }
}

TEST(ResultTest, BasicTypeTest) {
   
    Result<int, float> r0 = func0(0);
    EXPECT_TRUE(r0.is_ok());
    EXPECT_EQ(r0.ok_value(), 100);


    Result<int, float> r1 = func0(1);
    EXPECT_TRUE(r1.is_err());
    EXPECT_EQ(r1.err_value(), 3.14f);
}

TEST(ResultTest, ComplexTypeTest) {
    Ok<std::string> ok_str("ok");
    Result<std::string, std::string> ok_r(ok_str);
    EXPECT_TRUE(ok_r.is_ok());
    std::string value = ok_r.take_ok_value();
    EXPECT_TRUE(!value.empty());
    EXPECT_TRUE(ok_r.ok_value().empty());
}

TEST(ResultTest, MakeUtilityTest) {
    auto ok = make_ok(std::string("ok"));

    std::string err_str("err");
    auto err = make_err(err_str);
    
    std::string ok_str = ok.take_value();
    EXPECT_EQ(ok_str, std::string("ok"));
    EXPECT_TRUE(ok.value.empty());

    std::string err_str_take = err.take_value();
    EXPECT_EQ(err_str_take, err_str);
    EXPECT_TRUE(err.value.empty());
}
