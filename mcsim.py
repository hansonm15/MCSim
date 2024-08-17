'''Module to run a Monte Carlo Simulation'''

from collections import namedtuple
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import normalvariate
from typing import Callable, Generator

__all__ = [
    'InputVariable',
    'check_calculation',
    'run',
    'run_to_csv',
]

CaseVariable = namedtuple('CaseVariable', ['name', 'avg', 'std_dev'], )

@dataclass
class InputVariable:
    """Input variable to be used in the calculations.
    
    Attributes:
        name (str): Variable name, case-sensitive match to calculation inputs.
        case_avg (list[float]): List of averages used for each case. If only one case pass a singular list for cases (ex. [100])
        case_std_dev (list[float] | float | int): List of standard deviations used for each case. If standard deviation is the same (or zero for case values), use a float or int value.

    Yields:
        Generator[CaseVariable, None, None]: A namedtuple for a variable case.
    """    
    name: str
    case_avg: list[float]
    case_std_dev: list[float] | float | int
    
    def __post_init__(self) -> None:
        if isinstance(self.case_std_dev, float | int):
            self.cases = [CaseVariable(name=self.name, avg=avg, std_dev=self.case_std_dev) for avg in self.case_avg]
        else:
            self.cases = [CaseVariable(name=self.name, avg=avg, std_dev=std_dev) for avg, std_dev in zip(self.case_avg, self.case_std_dev)]
    
    def __iter__(self) -> Generator[CaseVariable, None, None]:
        for case in self.cases:
            yield case
    
    
def _gen_cases(inputs: list[InputVariable]) -> Generator[tuple[CaseVariable], None, None]:
    """Generates all possible case combinations.

    Args:
        inputs (list[InputVariable]): list of all Input Variables

    Yields:
        Generator[CaseVariable, None, None]: CaseVariables
    """    
    cases: list[tuple[CaseVariable]] = list(product(*inputs))
    for case in cases:
        yield case
    

def _gen_results(calculation: Callable, inputs: list[InputVariable], tests: int = 1) -> Generator[dict[str, float], None, None]:        
    """Generator to run all cases for given number of tests.

    Args:
        calculation (Callable): User-Defined function which returns dictionary as {OutputVarName: Value} 
        inputs (list[InputVariable]): Input variables to run in calculation
        tests (int, optional): Number of tests to run per case. Defaults to 1.

    Yields:
        Generator[dict[str, float], None, None]: Test result dictionary including inputs and outputs in format {VariableName: Value}
    """    
    for case in _gen_cases(inputs=inputs):
        for _ in range(tests):
            yield _test(
                calculation=calculation,
                inputs={var.name: normalvariate(mu=var.avg, sigma=var.std_dev) for var in case}
            )


def _test(calculation: Callable, inputs: dict[str, float]) -> dict[str, float]:
    """Run calculation test

    Args:
        calculation (Callable): User-Defined function which returns dictionary as {OutputVarName: Value} 
        inputs (dict[str, float]): Input variables to run in calculation

    Returns:
        dict[str, float]: Result dictionary including inputs and outputs in format {VariableName: Value}
    """    
    return inputs | calculation(**inputs)


def check_calculation(calculation: Callable, inputs: list[InputVariable]) -> dict[str, float]:
    """Runs calculation for checking function calculation. The inputs used will be the first case average for each InputVariable.

    Args:
        calculation (Callable): User-Defined function which returns dictionary as {OutputVarName: Value} 
        inputs (list[InputVariable]): Input variables to run in calculation

    Returns:
       dict[str, float]: Result dictionary including inputs and outputs in format {VariableName: Value}
    """    
    return _test(calculation=calculation, inputs={var.name: var.case_avg[0] for var in inputs})


def run(calculation: Callable, inputs: list[InputVariable], tests: int = 1) -> list[dict[str, float]]:
    """Runs all cases for given number of tests. Results are saved locally and can be turned into a Polars or Pandas DataFrame.

    Args:
        calculation (Callable): User-Defined function which returns dictionary as {OutputVarName: Value} 
        inputs (list[InputVariable]): Input variables to run in calculation
        tests (int, optional): Number of tests to run per case. Defaults to 1.

    Returns:
        list[dict[str, float]]: list of result dictionaries in the form of: {VariableName: Value}. Result dictionaries will include input variables and output variables
    """    
    return [
        _ for _ in _gen_results(calculation=calculation, inputs=inputs, tests=tests)
    ]

def run_to_csv(filepath: Path, calculation: Callable, inputs: list[InputVariable], tests: int=1) -> None:
    """Runs all cases for given number of tests. Inputs and outputs will be written to the given filepath in csv format.

    Args:
        filepath (Path): Filepath save location.
        calculation (Callable): User-Defined function which returns dictionary as {OutputVarName: Value} 
        inputs (list[InputVariable]): Input variables to run in calculation
        tests (int, optional): Number of tests to run per case. Defaults to 1.
    """    
    with open(filepath, 'w') as file:
        # Header
        file.write(
            ', '.join([name for name, _ in check_calculation(calculation, inputs).items()]) + '\n'
        )
        
        # Results
        for result in _gen_results(calculation=calculation, inputs=inputs, tests=tests):
            file.write(
                ', '.join([str(value) for _, value in result.items()]) + '\n'
            )


if __name__=='__main__':
    def func(var1: float, var2: float) -> dict[str, float]:
        """Test function

        Args:
            var1 (float): Input variable 1
            var2 (float): Input variable 2

        Returns:
            dict[str, float]: Output dictionary in form {VariableName: Value}
        """        
        return {'out1': var1/var2, 'out2': var1 + var2}
    
    # Inputs
    inputs = [
        InputVariable('var1', [1, 2, 3], 0),                # Case Values
        InputVariable('var2', [10, 11, 12], [1, 2, 3])      # Case Variables
    ]
    
    # Run MCS and save to file in csv format
    filepath = Path(__file__).parent.joinpath('test_file.csv')
    run_to_csv(filepath=filepath, calculation=func, inputs=inputs, tests=10)
    
    # Run MCS and store results locally
    r = run(calculation=func, inputs=inputs, tests=10)
    print(r)
