#ifndef PROGRESSBAR_9SHMI704
#define PROGRESSBAR_9SHMI704

class ProgressBar
{
    public:
        ProgressBar();
        void update(float ratio);
    private:
        int previousPercent;
};

#endif /* end of include guard: PROGRESSBAR_9SHMI704 */
