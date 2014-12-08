#include <mpi.h>
#include <iostream>
#include <sstream>

using namespace std;


int main(int argc, char* argv[])
{
    int my_rank; /* rang du processus */
    int p; /* nombre de processus */
    int source; /* rang de l’emetteur */
    int dest; /* rang du recepteur */

    int tag = 0; /* etiquette du message */
    char message[100];
    MPI_Status status;

    /* Initialisation */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    bool state = (my_rank == 0);
    int value = 0;

    while(true)
    {
        if (state)
        {
            /* Creation du message */
            char proc_name[100];
            int proc_length;
            MPI_Get_processor_name((char*)&proc_name,&proc_length);
            proc_name[proc_length+1] = (char)0;
            stringstream ss;
            ss << "Proc :" << my_rank;
            string s = ss.str();
            dest = (my_rank + 1);
            MPI_Send(s.c_str(), s.size()+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
            state = false;
            break;
        }
        else
        {
            MPI_Recv(message, 100, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            cout << message << "my rank = " << my_rank << endl;
            if (my_rank == 3)
                break;
            else
                state = true;
        }
    }
    /* Desactivation */
    MPI_Finalize();
    return 0;
}
