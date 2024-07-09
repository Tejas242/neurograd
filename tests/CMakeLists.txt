# Define the executable for testing
add_executable(test_neurograd test_neurograd.cpp)

# Link the library to the test executable
target_link_libraries(test_neurograd PRIVATE neurograd)

# Add the test
add_test(NAME neurograd_test COMMAND test_neurograd)
