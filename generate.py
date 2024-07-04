import sys
import copy

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for domain in self.domains.keys():
            wordsToRemoveFromDomain = []
            for word in self.domains[domain]:
                # check if length of word is correct - if not remove
                if len(word) != domain.length:
                    wordsToRemoveFromDomain.append(word)
            if len(wordsToRemoveFromDomain):
                for word in wordsToRemoveFromDomain:
                    self.domains[domain].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        toRemove = []
        # get x and y overlap
        overlap = self.crossword.overlaps[(x, y)]
        for varX in self.domains[x]:
            match = False
            if overlap is not None:
                for varY in self.domains[y]:
                    # check if x and y overlap in correct position
                    if varX[overlap[0]] == varY[overlap[1]]:
                        match = True
                if match == False:
                    toRemove.append(varX)
                    revised = True
        # remove all x not arc consistent with y
        if len(toRemove):
            for word in toRemove:
                if word in self.domains[x]:
                    self.domains[x].remove(word)
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = []
        # if no arcs get all from csp
        if arcs is None:
            for x in self.domains.keys():
                for y in self.domains.keys():
                    if x != y and self.crossword.overlaps[(x, y)] is not None:
                        queue.append((x, y))
        # else append arcs to queue
        else:
            for arc in arcs:
                queue.append(arc)
        # as long as there are arcs to be checked enforce arc consistency
        while len(queue) > 0:
            (X, Y) = queue.pop(0)
            if self.revise(X, Y):
                # check if domain is empty
                if len(self.domains[X]) == 0:
                    return False
                else:
                    for Z in self.crossword.neighbors(X):
                        if (Z != Y):
                            queue.append((Z, X))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for domain in self.domains:
            if domain not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        allValues = []
        for key in assignment:
            value = assignment[key]
            # check value are correct length
            if key.length != len(value):
                return False
            # check if value is distinct
            if value in allValues:
                return False
            else:
                allValues.append(value)
            # check for conflicts between neighbouring variables
            neighbours = self.crossword.neighbors(key)
            for neighbour in neighbours:
                overlap = self.crossword.overlaps[(key, neighbour)]
                if neighbour in assignment:
                    if assignment[neighbour][overlap[1]] != value[overlap[0]]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        listValues = {}
        for value in self.domains[var]:
            count = 0
            # get neighbours of var
            neighbours = self.crossword.neighbors(var)
            for neighbour in neighbours:
                # check if it is already assigned
                if neighbour in assignment:
                    continue
                # get overlap betweeb var and neighbour
                overlap = self.crossword.overlaps[(var, neighbour)]
                for y in self.domains[neighbour]:
                    if y[overlap[1]] != value[overlap[0]]:
                        count += 1
            listValues[value] = count
        # order the list of values by the var that rules out the fewest values
        return {k: v for k, v in sorted(listValues.items(), key=lambda item: item[1])}

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassignedVars = {}
        for key in self.domains.keys():
            # if variable is already assigned do not return it
            if key in assignment:
                continue
            else:
                degree = 0
                # get neighbours of the variable key
                neighbours = self.crossword.neighbors(key)
                degree = len(neighbours)
                unassignedVars[key] = -degree
        # sort the unnasigned variable based on the minimum number of remaining values in its domain and heighest degree
        if len(unassignedVars):
            domainsList = sorted(
                unassignedVars, key=lambda x: (len(self.domains[x]), unassignedVars[x]))
            return domainsList[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # check if assignment is complete
        if self.assignment_complete(assignment):
            return assignment
        # get all unnasigned values for assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            inferences = []
            # check assignment is consistent
            if self.consistent(assignment):
                # maintain arc consistency
                inferences = self.inferences(var, assignment)
                if inferences is not None:
                    for x in inferences:
                        assignment[x] = inferences[x]
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            # if tried value doesnt satisfy constraints - remove from assignemnt
            del assignment[var]
            if len(inferences):
                for x in inferences:
                    del assignment[x]
        return None

    def inferences(self, var, assignment):
        # deep copy assignment
        assignmentCopy = copy.deepcopy(assignment)
        # deep copy domains
        varCopy = copy.deepcopy(self.domains)
        # get neighbours for var
        neighbours = self.crossword.neighbors(var)
        # assign the current assignment values to their domains
        for a in assignment:
            self.domains[a] = [assignment[a]]
        arcs = []
        # get all the arcs based on neighbours
        for y in neighbours:
            arcs.append((y, var))
        if len(arcs) > 0:
            # check if arcs are arc consistent
            arcConsistent = self.ac3(arcs)
            if arcConsistent:
                # check for inferences
                inferences = {}
                for y in neighbours:
                    if y not in assignment:
                        # can be infered if neighbour has only 1 value left
                        if len(self.domains[y]) == 1:
                            inferences[y] = self.domains[y].pop()
                            assignmentCopy[y] = inferences[y]
                            # check for neighbour inferences
                            yInferences = self.inference(y, assignmentCopy)
                            if yInferences is not None:
                                # add neighbour inferences to all inferences
                                inferences.update(yInferences)
            # revert back the domains
            self.domains = varCopy
            if len(inferences.keys()):
                # if there are inferences return the list
                return inferences
            else:
                return None
        else:
            # revert back the domains
            self.domains = varCopy
            return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
