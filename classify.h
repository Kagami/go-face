typedef dlib::matrix<float,128,1> sample_type;

int classify(const std::vector<sample_type>& samples, const sample_type& test_sample);
